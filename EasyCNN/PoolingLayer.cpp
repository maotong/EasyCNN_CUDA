#include <algorithm>
#include <sstream>
#include "PoolingLayer.h"

#if WITH_OPENCV_DEBUG
#include "opencv2/opencv.hpp"
#endif

EasyCNN::PoolingLayer::PoolingLayer()
{

}

EasyCNN::PoolingLayer::~PoolingLayer()
{

}

void EasyCNN::PoolingLayer::setParamaters(const PoolingType _poolingType, const ParamSize _poolingKernelSize, const size_t _widthStep, const size_t _heightStep)
{
	easyAssert(_poolingKernelSize.number == 1 && _poolingKernelSize.channels > 0 && _poolingKernelSize.width > 1 && _poolingKernelSize.height > 1 && _widthStep > 0 && _heightStep > 0,
		"parameters invalidate.");

	poolingKernelSize = _poolingKernelSize;
	poolingType = _poolingType;
	widthStep = _widthStep;
	heightStep = _heightStep;
}

std::string EasyCNN::PoolingLayer::serializeToString() const
{
	const std::string spliter = " ";
	std::stringstream ss;

	//layer desc
	ss << getLayerType() << spliter
		<< poolingType << spliter
		<< poolingKernelSize.number << spliter
		<< poolingKernelSize.channels << spliter
		<< poolingKernelSize.width << spliter
		<< poolingKernelSize.height << spliter
		<< widthStep << spliter
		<< heightStep << spliter;

	return ss.str();
}

void EasyCNN::PoolingLayer::serializeFromString(const std::string content)
{
	std::stringstream ss(content);
	//layer desc
	std::string _layerType;
	int _poolingType = 0;
	ss >> _layerType
		>> _poolingType
		>> poolingKernelSize.number
		>> poolingKernelSize.channels
		>> poolingKernelSize.width
		>> poolingKernelSize.height
		>> widthStep
		>> heightStep;
	poolingType = (PoolingType)_poolingType;
	easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
	easyAssert((poolingType == MaxPooling || poolingType == MeanPooling), "pooling type is invalidate.");
	solveInnerParams();
}

DEFINE_LAYER_TYPE(EasyCNN::PoolingLayer, "PoolingLayer");
std::string EasyCNN::PoolingLayer::getLayerType() const
{
	return layerType;
}

void EasyCNN::PoolingLayer::solveInnerParams()
{
	easyAssert(poolingKernelSize.number > 0 && poolingKernelSize.channels > 0 && poolingKernelSize.width > 1 && poolingKernelSize.height > 1, "poolingKernelSize parameters invalidate.");
	const DataSize inputSize = getInputBucketSize();
	poolingKernelSize.number = 1;
	poolingKernelSize.channels = inputSize.channels;
	easyAssert(inputSize.number && poolingKernelSize.number && inputSize.channels == poolingKernelSize.channels &&
		inputSize.width > poolingKernelSize.width && inputSize.height > poolingKernelSize.height,
		"poolingKernelSize parameters invalidate.");
	DataSize outputSize;
	outputSize.number = inputSize.number;
	outputSize.channels = inputSize.channels;
	outputSize.width = (inputSize.width - poolingKernelSize.width) / widthStep + 1;
	outputSize.height = (inputSize.height - poolingKernelSize.height) / heightStep + 1;
	setOutpuBuckerSize(outputSize);

	if (getPhase() == Phase::Train && poolingType == PoolingType::MaxPooling)
	{
		maxIdxesBucket.reset(new ParamBucket(ParamSize(outputSize.number, outputSize.channels, outputSize.height, outputSize.width)));
	}
}

void EasyCNN::PoolingLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();

	const float* prevData = prevDataBucket->getData().get();
	float* nextData = nextDataBucket->getData().get();
	float* maxIdxes = nullptr;

	if (getPhase() == Phase::Train && poolingType == PoolingType::MaxPooling)
	{
		auto newSize = maxIdxesBucket->getSize();
		if (newSize.number != prevDataSize.number)
		{
			newSize.number = prevDataSize.number;
			maxIdxesBucket.reset(new ParamBucket(newSize));
		}
		maxIdxes = maxIdxesBucket->getData().get();
	}

	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{
			for (size_t nh = 0; nh < nextDataSize.height; nh++)
			{
				for (size_t nw = 0; nw < nextDataSize.width; nw++)
				{
					const size_t inStartX = nw * widthStep;
					const size_t inStartY = nh * heightStep;
					const size_t nextDataIdx = nextDataSize.getIndex(nn, nc, nh, nw);

					float result = 0;
					size_t maxIdx = 0;
					//MaxPooling
					if (poolingType == PoolingType::MaxPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDataIdx = prevDataSize.getIndex(nn, nc, inStartY + ph, inStartX + pw);
								if (result < prevData[prevDataIdx])
								{
									result = prevData[prevDataIdx];
									maxIdx = ph * poolingKernelSize.width + pw;
								}
							}
						}
						if (maxIdxes)
						{
							maxIdxes[nextDataIdx] = (float)maxIdx;
						}
					}
					//MeanPooling
					else if (poolingType == PoolingType::MeanPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDataIdx = prevDataSize.getIndex(nc, inStartY + ph, inStartX + pw);
								result += prevData[prevDataIdx];
							}
						}
						result /= poolingKernelSize.width * poolingKernelSize.height;
					}

					nextData[nextDataIdx] = result;
				}
			}
		}
	}
}

void EasyCNN::PoolingLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const DataSize nextDiffSize = nextDiffBucket->getSize();
	easyAssert(maxIdxesBucket->getSize()._3DSize() == nextDataSize._3DSize(), "idx size must equals with next data.");

	//update prevDiff data
	const float* maxIdxes = maxIdxesBucket->getData().get();
	const DataSize prevDiffSize(prevDataSize.number, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<DataBucket> prevDiffBucket(std::make_shared<DataBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);

	//calculate current inner diff 
	//none
	//pass next layer's diff to previous layer
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		const float* nextDiff = nextDiffBucket->getData().get() + pn * nextDiffSize._3DSize();
		float* prevDiff = prevDiffBucket->getData().get() + pn * prevDataSize._3DSize();

		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{
			for (size_t nh = 0; nh < nextDataSize.height; nh++)
			{
				for (size_t nw = 0; nw < nextDataSize.width; nw++)
				{
					const size_t inStartX = nw * widthStep;
					const size_t inStartY = nh * heightStep;
					const size_t nextDataIdx = nextDataSize.getIndex(nc, nh, nw);

					//MaxPooling
					if (poolingType == PoolingType::MaxPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDiffIdx = prevDataSize.getIndex(nc, inStartY + ph, inStartX + pw);
								if (ph * poolingKernelSize.width + pw == maxIdxes[nextDataIdx])
								{
									prevDiff[prevDiffIdx] += nextDiff[nextDataIdx];
								}
							}
						}
					}
					//MeanPooling
					else if (poolingType == PoolingType::MeanPooling)
					{
						const float meanDiff = nextDiff[nextDataIdx] / (float)(poolingKernelSize._2DSize());

						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDiffIdx = prevDataSize.getIndex(nc, inStartY + ph, inStartX + pw);
								prevDiff[prevDiffIdx] += meanDiff;
							}
						}
					}
				}
			}
		}
	}

	//update this layer's param
	//nop

	nextDiffBucket = prevDiffBucket;
}