package main

import (
	"fmt"
)

type (
	A interface {
		M1(B) C
	}
	B interface{}
	C interface{}
)

func NewAImpl() A {
	return &aImpl{}
}

type aImpl struct{}

func (ai *aImpl) M1(b B) C {
	return &cImpl{}
}

type cImpl struct{}

func (ci *cImpl) String() string {
	return "C impl"
}

func main() {
	a := NewAImpl()
	var b B = func(x int) int { return x * x }
	c := a.M1(b)

	if ci, ok := c.(fmt.Stringer); ok {
		fmt.Println(ci.String())
	} else {
		fmt.Println("Not a Stringer")
	}

	result := func(x int) int {
		switch {
		case x < 0:
			return -x
		case x == 0:
			return 1
		default:
			return x * x
		}
	}(2)
	fmt.Println("Result:", result)
}
