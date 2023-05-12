package main

import "C"
import (
	"encoding/json"
	"log"
)

type Tokenizer struct {
}

type EncodeArgs struct {
	Code string `json:"code"`
}

type DecodeArgs struct {
	Tokens []int `json:"tokens"`
}

type EncodeResult struct {
	Tokens []int `json:"tokens"`
}

type DecodeResult struct {
	Code string `json:"code"`
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{}
}

func (t *Tokenizer) Encode(args *EncodeArgs) *EncodeResult {
	return &EncodeResult{
		Tokens: []int{1, 2, 3},
	}
}

func (t *Tokenizer) Decode(args *DecodeArgs) *DecodeResult {
	return &DecodeResult{
		Code: "hello world",
	}
}

//export Encode
func Encode(encodeArgsJson *C.char) *C.char {
	encodeArgs := &EncodeArgs{}
	err := json.Unmarshal([]byte(C.GoString(encodeArgsJson)), encodeArgs)
	if err != nil {
		log.Println(err)
		return nil
	}

	encodeResult := NewTokenizer().Encode(encodeArgs)
	encodeResultJsonBytes, err := json.Marshal(encodeResult)
	if err != nil {
		log.Println(err)
		return nil
	}

	return C.CString(string(encodeResultJsonBytes))
}

//export Decode
func Decode(decodeArgsJson *C.char) *C.char {
	decodeArgs := &DecodeArgs{}
	err := json.Unmarshal([]byte(C.GoString(decodeArgsJson)), decodeArgs)
	if err != nil {
		log.Println(err)
		return nil
	}

	decodeResult := NewTokenizer().Decode(decodeArgs)
	decodeResultJsonBytes, err := json.Marshal(decodeResult)
	if err != nil {
		log.Println(err)
		return nil
	}

	return C.CString(string(decodeResultJsonBytes))
}

func main() {}
