#pragma once

#include "Configure.h"
#include "DataBucket.h"
#include "ParamBucket.h"

namespace EasyCNN
{
	class LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket) = 0;
		virtual std::shared_ptr<EasyCNN::DataBucket> getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket) = 0;
	};

	class CrossEntropyFunctor : public LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket);
		virtual std::shared_ptr<EasyCNN::DataBucket> getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket);
	};

	class MSEFunctor : public LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket);
		virtual std::shared_ptr<EasyCNN::DataBucket> getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
			const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket);
	};
}