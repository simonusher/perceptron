#include "pch.h"
#include "Adaline.h"

Adaline::Adaline(int numberOfInputs) : Perceptron(makeBipolar(numberOfInputs)) { }
