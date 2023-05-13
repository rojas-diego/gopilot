package main

import "C"
import (
	"encoding/json"
	"go/parser"
	"go/scanner"
	"go/token"
)

type ScanResult struct {
	Offsets  [][2]int `json:"offsets"`
	IDs      []int    `json:"ids"`
	Names    []string `json:"names"`
	Literals []string `json:"literals"`
}

//export Scan
func Scan(byteSequence *C.char) *C.char {
	sequence := C.GoString(byteSequence)
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(sequence))

	var s scanner.Scanner
	s.Init(file, []byte(sequence), nil, scanner.ScanComments)

	result := ScanResult{}

	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		result.Offsets = append(result.Offsets, [2]int{int(pos), int(pos) + len(lit)})
		result.IDs = append(result.IDs, int(tok))
		result.Names = append(result.Names, tok.String())
		result.Literals = append(result.Literals, lit)
	}

	// Marshal the result into JSON
	bytes, err := json.Marshal(result)
	if err != nil {
		return C.CString(err.Error())
	}

	return C.CString(string(bytes))
}

//export Parse
func Parse(byteSequence *C.char) *C.char {
	sequence := C.GoString(byteSequence)
	fset := token.NewFileSet()

	parsedFile, err := parser.ParseFile(fset, "", sequence, parser.ParseComments)
	if err != nil {
		return C.CString(err.Error())
	}

	// Marshal the result into JSON
	bytes, err := json.Marshal(parsedFile)
	if err != nil {
		return C.CString(err.Error())
	}
	return C.CString(string(bytes))
}

//export IDToTokenName
func IDToTokenName(id C.int) *C.char {
	return C.CString(token.Token(int(id)).String())
}

func main() {}
