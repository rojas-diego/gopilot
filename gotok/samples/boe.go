package main

import (
	"fmt"
)

type (
	Foo struct {
		field1 int
		field2 string
	}
	Bar func(Foo) bool
)

var (
	a = "hello" + " world"
	b = []int{1, 2, 3, 4}
)

func main() {
	foo := Foo{1, "text"}
	bar := Bar(func(f Foo) bool {
		return f.field1 > 0 && f.field2 != ""
	})

	if bar(foo) {
		fmt.Println(a, b)
	}

	for i, v := range []string{"a", "b", "c"}[1:] {
		fmt.Printf("%d: %s\n", i+1, v)
	}
}
