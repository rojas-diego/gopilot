package logger

import (
	"fmt"
	"sync/atomic"
)

var (
	InfoMessagePrefix string = "INFO: "
	ErrorMessagePrefix string = "ERROR: "
	WarnMessagePrefix string = "WARN: "
)

func