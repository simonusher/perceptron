#pragma once
#include "pch.h"
#include "Perceptron.h"
#include <random>
#include <iostream>
#include "Adaline.h"

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

class TrainingAlgorithm {
public:
	void train(Perceptron& neuron, const TrainingSet& trainingSet, double learningRateAlpha, int iterations, std::mt19937& randomGenerator);
	void train(Adaline& adaline, const TrainingSet& trainingSet, double learningRateAlpha, double learningStopErrorThreshold, std::mt19937& randomGenerator);
private:
	void initializePerceptron(Perceptron& neuron, std::mt19937& randomGenerator);
	void runTraining(Perceptron & neuron, const TrainingSet& trainingSet, double learningRateAlpha, int iterations);
	void trainOneIteration(Perceptron& neuron, const TrainingExample& example, double alpha);
	
	void runTrainingAdaline(Adaline& neuron, const TrainingSet& trainingSet, double learningRateAlpha, double errorThreshold);
	double trainOneIterationAdaline(Adaline& neuron, const TrainingSet& trainingSet, double alpha);
	
	double calculateError(double example, double actual);
};