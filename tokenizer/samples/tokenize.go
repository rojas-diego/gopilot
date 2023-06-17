package main

import "strings"

func Tokenize(src string) []string {
	return strings.Split(src, " ")
}
