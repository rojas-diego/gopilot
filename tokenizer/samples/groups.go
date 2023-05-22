package main

import (
	"context"
	"notes-service/auth"
	"notes-service/background"
	"notes-service/models"
	notesv1 "notes-service/protorepo/noted/notes/v1"
	"notes-service/validators"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type groupsAPI struct {
	notesv1.UnimplementedGroupsAPIServer

	logger *zap.Logger

	auth       auth.Service
	background background.Service

	groups     models.GroupsRepository
	activities models.ActivitiesRepository
	notes      models.NotesRepository
}

func (srv *groupsAPI) CreateGroup(ctx context.Context, req *notesv1.CreateGroupRequest) (*notesv1.CreateGroupResponse, error) {
	token, err := srv.authenticate(ctx)
	if err != nil {
		return nil, err
	}

	err = validators.ValidateCreateGroupRequest(req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	group, err := srv.groups.CreateGroup(ctx, &models.CreateGroupPayload{
		Name:        req.Name,
		Description: req.Description,
	}, token.AccountID)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	return &notesv1.CreateGroupResponse{Group: modelsGroupToProtobufGroup(group)}, nil
}

func (srv *groupsAPI) CreateWorkspace(ctx context.Context, req *notesv1.CreateWorkspaceRequest) (*notesv1.CreateWorkspaceResponse, error) {
	group, err := srv.groups.CreateWorkspace(ctx, &models.CreateWorkspacePayload{
		Name:           "My Workspace",
		Description:    "A space just for you",
		AvatarUrl:      "",
		OwnerAccountID: req.AccountId,
	}, req.AccountId)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	return &notesv1.CreateWorkspaceResponse{Group: modelsGroupToProtobufGroup(group)}, nil
}

func (srv *groupsAPI) GetGroup(ctx context.Context, req *notesv1.GetGroupRequest) (*notesv1.GetGroupResponse, error) {
	token, err := srv.authenticate(ctx)
	if err != nil {
		return nil, err
	}

	err = validators.ValidateGetGroupRequest(req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	group, err := srv.groups.GetGroupInternal(ctx, &models.OneGroupFilter{GroupID: req.GroupId})
	if err != nil {
		return nil, statusFromModelError(err)
	}

	// If user is a member or workspace owner, return the group.
	if group.FindMember(token.AccountID) != nil || (group.WorkspaceAccountID != nil && *group.WorkspaceAccountID == token.AccountID) {
		return &notesv1.GetGroupResponse{Group: modelsGroupToProtobufGroup(group)}, nil
	}

	// If user has been invited, return a public preview.
	if group.FindInviteByRecipient(token.AccountID) != nil {
		return &notesv1.GetGroupResponse{Group: modelsGroupToPublicProtobufGroup(group)}, nil
	}

	// If user has an invite code, return a public preview.
	if req.InviteLinkCode != "" && group.FindInviteLinkByCode(req.InviteLinkCode) != nil {
		return &notesv1.GetGroupResponse{Group: modelsGroupToPublicProtobufGroup(group)}, nil
	}

	// Otherwise user has not the right to access this group.
	return nil, status.Error(codes.PermissionDenied, "permission denied")
}

func (srv *groupsAPI) UpdateGroup(ctx context.Context, req *notesv1.UpdateGroupRequest) (*notesv1.UpdateGroupResponse, error) {
	token, err := srv.authenticate(ctx)
	if err != nil {
		return nil, err
	}

	err = validators.ValidateUpdateGroupRequest(req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	group, err := srv.groups.UpdateGroup(ctx, &models.OneGroupFilter{GroupID: req.GroupId}, &models.UpdateGroupPayload{
		Name:        req.Name,
		Description: req.Description,
	}, token.AccountID)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	return &notesv1.UpdateGroupResponse{Group: modelsGroupToProtobufGroup(group)}, nil
}

func (srv *groupsAPI) DeleteGroup(ctx context.Context, req *notesv1.DeleteGroupRequest) (*notesv1.DeleteGroupResponse, error) {
	token, err := srv.authenticate(ctx)
	if err != nil {
		return nil, err
	}

	err = validators.ValidateDeleteGroupRequest(req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	group, err := srv.groups.GetGroup(ctx, &models.OneGroupFilter{GroupID: req.GroupId}, token.AccountID)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	// Change the note's group to the first user's workspace if it has one
	for _, member := range *group.Members {
		err = srv.moveNotesToUserWorkspaceOrDeleteThem(ctx,
			&models.ManyNotesFilter{AuthorAccountID: member.AccountID, GroupID: req.GroupId},
		)
		if err != nil {
			srv.logger.Error("could not move notes: " + err.Error())
		}
	}

	err = srv.groups.DeleteGroup(ctx, &models.OneGroupFilter{GroupID: req.GroupId}, token.AccountID)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	return &notesv1.DeleteGroupResponse{}, nil
}

func (srv *groupsAPI) ListGroups(ctx context.Context, req *notesv1.ListGroupsRequest) (*notesv1.ListGroupsResponse, error) {
	token, err := srv.authenticate(ctx)
	if err != nil {
		return nil, err
	}

	err = validators.ValidateListGroupsRequest(req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	groups, err := srv.groups.ListGroupsInternal(ctx, &models.ManyGroupsFilter{AccountID: token.AccountID}, nil)
	if err != nil {
		return nil, statusFromModelError(err)
	}

	return &notesv1.ListGroupsResponse{Groups: modelsGroupsToProtobufGroups(groups)}, nil
}

func (srv *groupsAPI) authenticate(ctx context.Context) (*auth.Token, error) {
	token, err := srv.auth.TokenFromContext(ctx)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, "invalid token")
	}
	return token, nil
}

func modelsGroupToPublicProtobufGroup(group *models.Group) *notesv1.Group {
	return &notesv1.Group{
		Id:          group.ID,
		Name:        group.Name,
		Description: group.Description,
		AvatarUrl:   group.AvatarUrl,
		CreatedAt:   timestamppb.New(group.CreatedAt),
		ModifiedAt:  timestamppb.New(group.ModifiedAt),
	}
}

func modelsGroupsToProtobufGroups(groups []*models.Group) []*notesv1.Group {
	protoGroups := make([]*notesv1.Group, len(groups))

	for i := range groups {
		protoGroups[i] = modelsGroupToProtobufGroup(groups[i])
	}

	return protoGroups
}

func modelsGroupToProtobufGroup(group *models.Group) *notesv1.Group {
	var members []*notesv1.GroupMember = nil
	var invites []*notesv1.GroupInvite = nil
	var inviteLinks []*notesv1.GroupInviteLink = nil
	var conversations []*notesv1.GroupConversation = nil
	/*activities*/

	if group.Members != nil {
		members = make([]*notesv1.GroupMember, len(*group.Members))
		for i := range *group.Members {
			members[i] = &notesv1.GroupMember{
				AccountId: (*group.Members)[i].AccountID,
				JoinedAt:  timestamppb.New((*group.Members)[i].JoinedAt),
				IsAdmin:   (*group.Members)[i].IsAdmin,
			}
		}
	}

	if group.Invites != nil {
		invites = make([]*notesv1.GroupInvite, len(*group.Invites))
		for i := range *group.Invites {
			invites[i] = &notesv1.GroupInvite{
				Id:                 (*group.Invites)[i].ID,
				RecipientAccountId: (*group.Invites)[i].RecipientAccountID,
				SenderAccountId:    (*group.Invites)[i].SenderAccountID,
				CreatedAt:          timestamppb.New((*group.Invites)[i].CreatedAt),
				ValidUntil:         timestamppb.New((*group.Invites)[i].ValidUntil),
			}
		}
	}

	if group.InviteLinks != nil {
		inviteLinks = make([]*notesv1.GroupInviteLink, len(*group.InviteLinks))
		for i := range *group.InviteLinks {
			inviteLinks[i] = &notesv1.GroupInviteLink{
				Code:                 (*group.InviteLinks)[i].Code,
				GeneratedByAccountId: (*group.InviteLinks)[i].GeneratedByAccountID,
				CreatedAt:            timestamppb.New((*group.InviteLinks)[i].CreatedAt),
				ValidUntil:           timestamppb.New((*group.InviteLinks)[i].ValidUntil),
			}
		}
	}

	if group.Conversations != nil {
		conversations = make([]*notesv1.GroupConversation, len(*group.Conversations))
		for i := range *group.Conversations {
			conversations[i] = &notesv1.GroupConversation{
				Id:        (*group.Conversations)[i].ID,
				Name:      (*group.Conversations)[i].Name,
				CreatedAt: timestamppb.New((*group.Conversations)[i].CreatedAt),
			}
		}
	}

	workspaceAccountID := ""

	if group.WorkspaceAccountID != nil {
		workspaceAccountID = *group.WorkspaceAccountID
	}

	return &notesv1.Group{
		Id:                 group.ID,
		Name:               group.Name,
		Description:        group.Description,
		WorkspaceAccountId: workspaceAccountID,
		AvatarUrl:          group.AvatarUrl,
		CreatedAt:          timestamppb.New(group.CreatedAt),
		ModifiedAt:         timestamppb.New(group.ModifiedAt),
		Members:            members,
		Conversations:      conversations,
		Invites:            invites,
		InviteLinks:        inviteLinks,
	}
}
