#include "Perceptron.h"
#include <utility>

Perceptron::Perceptron(int numberOfInputs, std::function<double(double)> transitionFunction) :
	weights(numberOfInputs), transitionFunction(std::move(transitionFunction)) { }

double Perceptron::forward(const valarray<double>& inputs) const {
	return (weights * inputs).sum();
}

double Perceptron::calculateOutput(const valarray<double>& inputs) const {
	return transitionFunction(forward(inputs));
}

void Perceptron::learn(const valarray<double>& deltas) {
	weights += deltas;
}

void Perceptron::initializeRandomly(std::mt19937& randomGenerator, double lowerThreshold, double upperThreshold) {
	std::uniform_real_distribution<double> distribution(lowerThreshold, upperThreshold);
	for(int i = 0; i < weights.size(); i++) {
		weights[i] = distribution(randomGenerator);
	}
}

void Perceptron::initialize(const valarray<double>& weights) {
	this->weights = weights;
}

int Perceptron::getNumberOfInputs() const {
	return weights.size();
}

valarray<double> Perceptron::getWeights() const {
	return weights;
}

Perceptron Perceptron::makeUnipolar(int numberOfInputs) {
	return Perceptron(numberOfInputs, [](double activation) { return activation > 0.5 ? 1 : 0; });
}

Perceptron Perceptron::makeBipolar(int numberOfInputs) {
	return Perceptron(numberOfInputs, [](double activation) { return activation > 0 ? 1 : -1; });
}


std::ostream& operator<<(std::ostream& os, const Perceptron& obj) {
	os << "Perceptron \n\tInputs: " << obj.getNumberOfInputs() << ""
		"\n\tWeights: [ ";
	for (int i = 0; i < obj.weights.size(); i++) {
		os << obj.weights[i];
		if (i != obj.weights.size() - 1) {
			os << ", ";
		}
	}
	os << " ]" << std::endl;
	return os;
}
