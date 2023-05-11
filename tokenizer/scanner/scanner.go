package main

import (
	"encoding/json"
	"fmt"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"log"
	"os"
)

const (
	TAB = int(token.TILDE + 1 + iota)
	NEWLINE
	SPACE
)

type TokenizerOutput struct {
	TokenIDs    []int    `json:"token_ids"`
	TokenNames  []string `json:"token_names"`
	TokenValues []string `json:"token_values"`
}

func (t *TokenizerOutput) Append(id int, name, value string) {
	t.TokenIDs = append(t.TokenIDs, id)
	t.TokenNames = append(t.TokenNames, name)
	t.TokenValues = append(t.TokenValues, value)
}

type ScanResult struct {
	Pos token.Pos
	Tok token.Token
	Lit string
}

func main() {
	rawContents, tokens, fset := scanAllFromStdinOrPanic()
	result := TokenizerOutput{
		TokenIDs:    []int{},
		TokenNames:  []string{},
		TokenValues: []string{},
	}

	// Post-process the scanned tokens to handle whitespaces and normalize the
	// output.
	previousTokenEndPos := 0
	for i, tok := range tokens {
		tokenPos := fset.Position(tok.Pos)
		tokenStartPos := tokenPos.Offset
		tokenValue := tok.Lit
		// Some tokens don't have a literal value. Hence we need to calculate
		// the end position of the token by hand.
		if tokenValue == "" {
			tokenValue = tok.Tok.String()
		}
		tokenEndPos := tokenStartPos + len(tokenValue)

		// Sometimes, the scanner produces tokens that are implicit in the
		// source code. We musn't include these tokens in the output. We do a
		// lookup of the next token to see if its start/end position overlap
		// with the next token. If so, we skip the current token.
		if i+1 < len(tokens) {
			nextToken := tokens[i+1]
			nextTokenPos := fset.Position(nextToken.Pos)
			// log.Printf("Token: %s, Next Token: %s", tok.Tok.String(), nextToken.Tok.String())
			// Check for overlap
			if tokenStartPos >= nextTokenPos.Offset {
				// log.Printf("Skipping implicit token: %s", tok.Tok.String())
				continue
			}
		}

		// Debug print the token start and end position
		// fmt.Printf("%s\t\t(%3d:%3d)\t%q\n", tok.Tok.String(), tokenStartPos, tokenEndPos, tok.Lit)

		// If the token does not immediately follow the last token, this means
		// some whitespace is present in between tokens[i] and tokens[i-1]
		if i > 0 && tokenStartPos != previousTokenEndPos {
			whitespace := string(rawContents[previousTokenEndPos:tokenStartPos])

			// Let's breakdown this unexpected whitespace into individual
			// whitespace tokens.
			// TODO: Consider 4 spaces as a tab, etc...
			for _, char := range whitespace {
				switch char {
				case '\n':
					result.Append(NEWLINE, "NEWLINE", "\n")
				case '\t':
					result.Append(TAB, "TAB", "\t")
				case ' ':
					result.Append(SPACE, "SPACE", " ")
				case '\r':
					// ignore carriage return
				default:
					log.Fatalf("Unexpected whitespace character: %c", char)
				}
			}
		}

		result.Append(int(tok.Tok), tok.Tok.String(), tokenValue)
		previousTokenEndPos = tokenEndPos
	}

	// Output prettified JSON
	bytes, _ := json.MarshalIndent(result, "", "  ")
	fmt.Printf("%+v\n", string(bytes))
}

// Reads all contents from stdin and returns a slice of ScanResults
// representing all tokens in the file up to EOF.
func scanAllFromStdinOrPanic() (string, []ScanResult, *token.FileSet) {
	contents, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(contents))
	var s scanner.Scanner
	s.Init(file, []byte(contents), nil, scanner.ScanComments)
	tokens := []ScanResult{}
	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		tokens = append(tokens, ScanResult{pos, tok, lit})
	}
	return string(contents), tokens, fset
}
