// perceptron.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Perceptron.h"
#include "TrainingAlgorithm.h"
#include <random>
#include "TestRunner.h"

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

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

	std::vector<std::pair<std::valarray<double>, double>> orBias{
		{ { 0, 0 }, 0 },
		{ { 0, 1 }, 1 },
		{ { 1, 0 }, 1 },
		{ { 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> andBias{
		{ { 1, 0, 0 }, 0 },
		{ { 1, 0, 1 }, 0 },
		{ { 1, 1, 0 }, 0 },
		{ { 1, 1, 1 }, 1 },
	};


	std::vector<std::pair<std::valarray<double>, double>> orBipolar{
		{ { -1, -1 }, -1 },
		{ { -1, 1 }, 1 },
		{ { 1, -1 }, 1 },
		{ { 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> andBipolar{
		{ { -1, -1 }, -1 },
		{ { -1, 1 }, -1 },
		{ { 1, -1 }, -1 },
		{ { 1, 1 }, 1 },
	};
	

	std::vector<std::pair<std::valarray<double>, double>> orTrainingSetWithBias {
		{ { 1, -1, -1 }, -1 },
		{ { 1, -1, 1 }, 1 },
		{ { 1, 1, -1 }, 1 },
		{ { 1, 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> andTrainingSetWithBias {
		{ { 1, -1, -1 }, -1 },
		{ { 1, -1, 1 }, -1 },
		{ { 1, 1, -1 }, -1 },
		{ { 1, 1, 1 }, 1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> nandTrainingSetWithBias{
	{ { 1, -1, -1 }, 1 },
	{ { 1, -1, 1 }, 1 },
	{ { 1, 1, -1 }, 1 },
	{ { 1, 1, 1 }, -1 },
	};

	std::vector<std::pair<std::valarray<double>, double>> norTrainingSetWithBias{
	{ { 1, -1, -1 }, 1 },
	{ { 1, -1, 1 }, -1 },
	{ { 1, 1, -1 }, -1 },
	{ { 1, 1, 1 }, -1 },
	};
	
	std::random_device randomDevice;
	std::mt19937 randomGenerator(randomDevice());


	// std::vector<TrainingSet> trainingSets{
	// 	orTrainingSetWithBias, andTrainingSetWithBias
	// };

	std::vector<TrainingSet> unipolar{
		orBias, andBias
	};

	std::vector<TrainingSet> bipolar{
		orBipolar, andBipolar
	};
	
	std::vector<TrainingSet> trainingSets{
		orTrainingSetWithBias, andTrainingSetWithBias
	};

	std::vector<std::pair<double, double>> ranges {
		{-2, 2 },
		{-1, 1 },
		{ -0.8, 0.8 },
		{ -0.5, 0.5 },
		{ -0.2, 0.2 },
		{ -0.1, 0.1 },
		{ -0.05, 0.05 },
		{ -0.01, 0.01 },
		{ -0.001, 0.001 },
	};

	// testRunner::runAdalineRandomInitializationTests(ranges, trainingSets, 0.05, 0.3);
	// testRunner::runPerceptronRandomInitializationTests(ranges, trainingSets, 0.05);
	//

	std::vector<double> alphas{ 1e-6, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9};
	std::pair<double, double> range{ -0.1, 0.1 };
	
	// testRunner::runAdalineAlphaTests(alphas, trainingSets, range, 0.3);
	// testRunner::runPerceptronAlphaTests(alphas, trainingSets, range);

	testRunner::runUnipolarTest(0.05, unipolar, range);
	testRunner::runBipolarTest(0.05, trainingSets, range);
}

