#pragma once

#include <string>

#include "Configure.h"

namespace EasyCNN
{
	void setAssertFatalCallback(void(*cb)(void* userData, const std::string& errorStr), void* userData);

	void easyAssertCore(const std::string& file, const std::string& function, const long line,
		const bool condition, const char* fmt, ...);

#define easyAssert(condition,fmt,...) \
	easyAssertCore(__FILE__, __FUNCTION__, __LINE__, (condition), (fmt), ##__VA_ARGS__);
}