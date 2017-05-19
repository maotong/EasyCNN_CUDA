#include <iostream>
#include <cassert>
#include <algorithm>

#include "EasyCNN.h"
#include "mnistDataLoader.h"


const int classes = 10;

static EasyCNN::NetWork buildConvNet(const size_t batch, const size_t channels, const size_t width, const size_t height)
{
	EasyCNN::NetWork network;
	network.setPhase(EasyCNN::Phase::Train);
	network.setInputSize(EasyCNN::DataSize(batch, channels, width, height));
	network.setLossFunctor(std::make_shared<EasyCNN::CrossEntropyFunctor>());

	//input data layer 0
	std::shared_ptr<EasyCNN::InputLayer> _0_inputLayer(std::make_shared<EasyCNN::InputLayer>());
	network.addLayer(_0_inputLayer);

	//convolution layer 1
	std::shared_ptr<EasyCNN::ConvolutionLayer> _1_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_1_convLayer->setParamaters(EasyCNN::ParamSize(6, 1, 5, 5), 1, 1, true);
	network.addLayer(_1_convLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//pooling layer 2
	std::shared_ptr<EasyCNN::PoolingLayer> _2_poolingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_2_poolingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 6, 2, 2), 2, 2);
	network.addLayer(_2_poolingLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//convolution layer 3
	std::shared_ptr<EasyCNN::ConvolutionLayer> _3_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_3_convLayer->setParamaters(EasyCNN::ParamSize(16, 6, 5, 5), 1, 1, true);
	network.addLayer(_3_convLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//pooling layer 4
	std::shared_ptr<EasyCNN::PoolingLayer> _4_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_4_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 16, 2, 2), 2, 2);
	network.addLayer(_4_pooingLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//full connect layer 5
	std::shared_ptr<EasyCNN::FullconnectLayer> _5_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_5_fullconnectLayer->setParamaters(EasyCNN::ParamSize(1, 512, 1, 1), true);
	network.addLayer(_5_fullconnectLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//full connect layer 6
	std::shared_ptr<EasyCNN::FullconnectLayer> _6_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_6_fullconnectLayer->setParamaters(EasyCNN::ParamSize(1, classes, 1, 1), true);
	network.addLayer(_6_fullconnectLayer);
	network.addLayer(std::make_shared<EasyCNN::ReluLayer>());

	//soft max layer 7
	std::shared_ptr<EasyCNN::SoftmaxLayer> _7_softmaxLayer(std::make_shared<EasyCNN::SoftmaxLayer>());
	network.addLayer(_7_softmaxLayer);

	return network;
}

//fetch data
static bool fetch_data(const std::vector<image_t>& images, std::shared_ptr<EasyCNN::DataBucket> inputDataBucket,
	const std::vector<label_t>& labels, std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const size_t offset, const size_t length)
{
	assert(images.size() == labels.size() && inputDataBucket->getSize().number == labelDataBucket->getSize().number);

	if (offset >= images.size())
	{
		return false;
	}

	size_t actualEndPos = offset + length;
	if (actualEndPos > images.size())
	{
		//image data
		auto inputDataSize = inputDataBucket->getSize();
		inputDataSize.number = images.size() - offset;
		actualEndPos = offset + inputDataSize.number;
		inputDataBucket.reset(new EasyCNN::DataBucket(inputDataSize));

		//label data
		auto labelDataSize = labelDataBucket->getSize();
		labelDataSize.number = inputDataSize.number;
		labelDataBucket.reset(new EasyCNN::DataBucket(inputDataSize));
	}

	//copy
	const size_t sizePerImage = inputDataBucket->getSize()._3DSize();
	const size_t sizePerLabel = labelDataBucket->getSize()._3DSize();
	assert(sizePerImage == images[0].channels * images[0].width * images[0].height);

	//scale to 0.0f~1.0f
	const float scaleRate = 1.0f / 256.0f;
	for (size_t i = offset; i < actualEndPos; i++)
	{
		float* inputData = inputDataBucket->getData().get() + (i - offset) * sizePerImage;
		const uint8_t* imageData = &images[i].data[0];
		for (size_t j = 0; j < sizePerImage; j++)
		{
			inputData[j] = (float)imageData[j] * scaleRate;
		}

		//label data
		float* labelData = labelDataBucket->getData().get() + (i - offset) * sizePerLabel;
		const uint8_t label = labels[i].data;
		for (size_t j = 0; j < sizePerLabel; j++)
		{
			if (j == label)
			{
				labelData[j] = 1.0f;
			}
			else
			{
				labelData[j] = 0.0f;
			}
		}
	}
	return true;
}

static std::shared_ptr<EasyCNN::DataBucket> convertVectorToDataBucket(const std::vector<image_t>& test_images, const size_t start, const size_t len)
{
	assert(test_images.size() > 0);
	const size_t number = len;
	const size_t channel = test_images[0].channels;
	const size_t width = test_images[0].width;
	const size_t height = test_images[0].height;
	const size_t sizePerImage = channel * width * height;
	const float scaleRate = 1.0f / 256.0f;
	std::shared_ptr<EasyCNN::DataBucket> result(new EasyCNN::DataBucket(EasyCNN::DataSize(number, channel, width, height)));
	for (size_t i = start; i < start + len; i++)
	{
		//image data
		float* inputData = result->getData().get() + (i - start) * sizePerImage;
		const uint8_t* imageData = &test_images[i].data[0];
		for (size_t j = 0; j < sizePerImage; j++)
		{
			inputData[j] = (float)imageData[j] * scaleRate;
		}
	}
	return result;
}

static uint8_t getMaxIdxInArray(const float* start, const float* stop)
{
	assert(start && stop && stop >= start);
	ptrdiff_t result = 0;
	const ptrdiff_t len = stop - start;
	for (ptrdiff_t i = 0; i < len; i++)
	{
		if (start[i] >= start[result])
		{
			result = i;
		}
	}
	return (uint8_t)result;
}

static float test(EasyCNN::NetWork& network, const size_t batch, const std::vector<image_t>& test_images, const std::vector<label_t>& test_labels)
{
	assert(test_images.size() == test_labels.size() && test_images.size()>0);

	int correctCount = 0;

	for (size_t i = 0; i < test_labels.size(); i += batch)
	{
		const size_t start = i;
		const size_t len = std::min(test_labels.size() - start, batch);
		const std::shared_ptr<EasyCNN::DataBucket> inputDataBucket = convertVectorToDataBucket(test_images, start, len);
		const std::shared_ptr<EasyCNN::DataBucket> probDataBucket = network.testBatch(inputDataBucket);
		const size_t labelSize = probDataBucket->getSize()._3DSize();
		const float* probData = probDataBucket->getData().get();
		for (size_t j = 0; j < len; j++)
		{
			const uint8_t stdProb = test_labels[i + j].data;
			const uint8_t testProb = getMaxIdxInArray(probData + j * labelSize, probData + (j + 1) * labelSize);
			if (stdProb == testProb)
			{
				correctCount++;
			}
		}
	}
	const float result = (float)correctCount / (float)test_labels.size();

	return result;
}


//image shuffle using random_shuffle in algorithm
static void shuffle_data(std::vector<image_t>& images, std::vector<label_t>& labels)
{
	assert(images.size() == labels.size());
	std::vector<size_t> indexArray;
	for (size_t i = 0; i < images.size(); i++)
	{
		indexArray.push_back(i);
	}
	std::random_shuffle(indexArray.begin(), indexArray.end());

	std::vector<image_t> tmpImages(images.size());
	std::vector<label_t> tmpLabels(labels.size());
	for (size_t i = 0; i < images.size(); i++)
	{
		const size_t srcIndex = i;
		const size_t dstIndex = indexArray[i];
		tmpImages[srcIndex] = images[dstIndex];
		tmpLabels[srcIndex] = labels[dstIndex];
	}
	images = tmpImages;
	labels = tmpLabels;
}


static void train(const std::string& mnist_train_images_file, const std::string& mnist_train_labels_file)
{
	bool success = false;

	EasyCNN::setLogLevel(EasyCNN::EASYCNN_LOG_LEVEL_CRITICAL);

	//load train images
	EasyCNN::logCritical("loading training data...");
	std::vector<image_t> images;
	success = load_mnist_images(mnist_train_images_file, images);
	assert(success && images.size() > 0);
	//load train labels
	std::vector<label_t> labels;
	success = load_mnist_labels(mnist_train_labels_file, labels);
	assert(success && labels.size() > 0);
	assert(images.size() == labels.size());

	shuffle_data(images, labels);

	//train data & validate data sparated.3:1
	//train
	std::vector<image_t> train_images(static_cast<size_t>(images.size() * 0.75f));
	std::vector<label_t> train_labels(static_cast<size_t>(labels.size() * 0.75f));
	std::copy(images.begin(), images.begin() + train_images.size(), train_images.begin());
	std::copy(labels.begin(), labels.begin() + train_labels.size(), train_labels.begin());
	//validate
	std::vector<image_t> validate_images(images.size() - train_images.size());
	std::vector<label_t> validate_labels(labels.size() - train_labels.size());
	std::copy(images.begin() + train_images.size(), images.end(), validate_images.begin());
	std::copy(labels.begin() + train_labels.size(), labels.end(), validate_labels.begin());
	EasyCNN::logCritical("load training data done. train set's size is %d,validate set's size is %d", train_images.size(), validate_images.size());

	//configuration
	float learningRate = 0.1f;
	const float decayRate = 0.001f;
	const float minLearningRate = 0.001f;

	const size_t testAfterBatches = 200;
	const size_t maxBatches = 10000;
	const size_t max_epoch = 4;
	const size_t batch = 16;
	const size_t channels = images[0].channels;
	const size_t width = images[0].width;
	const size_t height = images[0].height;

	EasyCNN::logCritical("max_epoch:%d, testAfterBatches:%d", max_epoch, testAfterBatches);
	EasyCNN::logCritical("learningRate:%f ,decayRate:%f , minLearningRate:%f", learningRate, decayRate, minLearningRate);
	EasyCNN::logCritical("channels:%d , width:%d , height:%d", channels, width, height);

	EasyCNN::logCritical("construct network begin...");
	EasyCNN::NetWork network(buildConvNet(batch, channels, width, height));
	EasyCNN::logCritical("construct network done.");

	//train
	EasyCNN::logCritical("begin training...");
	std::shared_ptr<EasyCNN::DataBucket> inputDataBucket = std::make_shared<EasyCNN::DataBucket>(EasyCNN::DataSize(batch, channels, width, height));
	std::shared_ptr<EasyCNN::DataBucket> labelDataBucket = std::make_shared<EasyCNN::DataBucket>(EasyCNN::DataSize(batch, classes, 1, 1));
	size_t epochIdx = 0;

	while (epochIdx < max_epoch)
	{
		size_t batchIdx = 0;
		while (true)
		{
			if (!fetch_data(train_images, inputDataBucket, train_labels, labelDataBucket, batchIdx * batch, batch))
			{
				break;
			}

			const float loss = network.trainBatch(inputDataBucket, labelDataBucket, learningRate);

			if (batchIdx > 0 && batchIdx % testAfterBatches == 0)
			{
				learningRate -= decayRate;
				learningRate = std::max(learningRate, minLearningRate);
				const float accuracy = test(network, 128, validate_images, validate_labels);
				EasyCNN::logCritical("sample : %d/%d , learningRate : %f , loss : %f , accuracy : %.4f%%",
					batchIdx * batch, train_images.size(), learningRate, loss, accuracy * 100.0f);
			}
			if (batchIdx >= maxBatches)
			{
				break;
			}
			batchIdx++;
		}
		if (batchIdx >= maxBatches)
		{
			break;
		}
		const float accuracy = test(network, 128, validate_images, validate_labels);
		EasyCNN::logCritical("epoch[%d] accuracy : %.4f%%", epochIdx++, accuracy * 100.0f);
	}
	const float accuracy = test(network, 128, validate_images, validate_labels);
	EasyCNN::logCritical("final accuracy : %.4f%%", accuracy * 100.0f);
	//success = network.saveModel(modelFilePath);
	//assert(success);
	EasyCNN::logCritical("finished training.");

}

int main(int argc, char* argv[])
{
	//mnist_date file path
	const std::string mnist_train_images_file = "mnist_data/train-images.idx3-ubyte";
	const std::string mnist_train_labels_file = "mnist_data/train-labels.idx1-ubyte";
	train(mnist_train_images_file, mnist_train_labels_file);
	system("pause");

	return 0;
}