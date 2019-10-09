// perceptron.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "Perceptron.h"
#include "TrainingAlgorithm.h"
#include <random>

int main()
{


	std::vector<std::pair<std::valarray<double>, double>> orTrainingSet {
		{ { 0, 0 }, 0 },
		{ { 0, 1 }, 1 },
		{ { 1, 0 }, 1 },
		{ { 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> andTrainingSet {
		{ { 0, 0 }, 0 },
		{ { 0, 1 }, 0 },
		{ { 1, 0 }, 0 },
		{ { 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> orTrainingSetWithBias {
		{ { 1, 0, 0 }, -1 },
		{ { 1, 0, 1 }, 1 },
		{ { 1, 1, 0 }, 1 },
		{ { 1, 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> andTrainingSetWithBias {
		{ { 1, 0, 0 }, -1 },
		{ { 1, 0, 1 }, -1 },
		{ { 1, 1, 0 }, -1 },
		{ { 1, 1, 1 }, 1 },
	};
	
	std::random_device randomDevice;
	std::mt19937 randomGenerator(randomDevice());


	Adaline p(3);
	TrainingAlgorithm l;
	std::vector<std::pair<std::valarray<double>, double>>& trainingSet = orTrainingSetWithBias;
	// std::vector<std::pair<std::valarray<double>, double>>& trainingSet = orTrainingSet;
	
	p.initializeRandomly(randomGenerator, -0.1, 0.1);
	
	l.train(p, trainingSet, 0.01, 0.3);
	// l.train(p, trainingSet, 0.01, 300);


	std::cout << "LEARNED: " << std::endl;
	for (int i = 0; i < trainingSet.size(); i++) {
		std::cout << p.calculateOutput(trainingSet[i].first) << std::endl;
	}

	std::cout << p;
}

