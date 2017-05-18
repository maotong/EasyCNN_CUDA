#pragma once

#include <memory>
#include <string>
#include "Configure.h"
#include "DataBucket.h"
#include "ParamBucket.h"

#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type,type_string) const std::string class_type::layerType = type_string; 
#define FRIEND_WITH_NETWORK friend class NetWork;

namespace EasyCNN
{
	enum class Phase
	{
		Train,
		Test
	};

	class Layer
	{
		FRIEND_WITH_NETWORK
	protected:
		virtual std::string getLayerType() const = 0;
		virtual std::string serializeToString() const{ return getLayerType(); };
		virtual void serializeFromString(const std::string content){/*nop*/ };
		//phase
		inline void setPhase(Phase phase) { this->phase = phase; }
		inline Phase getPhase() const{ return phase; }
		//learning rate
		inline void setLearningRate(const float learningRate){ this->learningRate = learningRate; }
		inline float getLearningRate() const{ return learningRate; }
		//size
		inline void setInputBucketSize(const DataSize size){ inputSize = size; }
		inline DataSize getInputBucketSize() const{ return inputSize; }
		inline void setOutpuBuckerSize(const DataSize size){ outputSize = size; }
		inline DataSize getOutputBucketSize() const{ return outputSize; }
		//solve params
		virtual void solveInnerParams(){ outputSize = inputSize; }
		//data flow		
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) = 0;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket) = 0;
	private:
		Phase phase = Phase::Train;
		DataSize inputSize;
		DataSize outputSize;
		float learningRate = 0.1f;
	};
}