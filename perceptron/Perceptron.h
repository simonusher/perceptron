#pragma once
#include <vector>
#include <functional>
#include <valarray>
#include <ostream>
#include <random>

using std::vector;
using std::valarray;

class Perceptron {
public:
	Perceptron(int numberOfInputs, std::function<double(double)> transitionFunction);
	double calculateOutput(const valarray<double>& inputs) const;
	void learn(const valarray<double>& deltas);
	void initializeRandomly(std::mt19937& randomGenerator, double lowerThreshold, double upperThreshold);
	void initialize(const valarray<double>& weights);
	int getNumberOfInputs() const;
	valarray<double> getWeights() const;

	friend std::ostream& operator<<(std::ostream& os, const Perceptron& obj);
	friend class TrainingAlgorithm;
	static Perceptron makeUnipolar(int numberOfInputs);
	static Perceptron makeBipolar(int numberOfInputs);
private:
	double forward(const valarray<double>& inputs) const;
	valarray<double> weights;
	std::function<double(double)> transitionFunction;
};