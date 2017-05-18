#include <cassert>
#include <cstdlib>
#include "EasyAssert.h"
#include "EasyLogger.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

static void* globalAssertFatalUserData = nullptr;
static void(*globalAssertFatalCB)(void* userData, const std::string& errorStr) = nullptr;

void EasyCNN::setAssertFatalCallback(void(*cb)(void* userData, const std::string& errorStr), void* userData)
{
	globalAssertFatalCB = cb;
	globalAssertFatalUserData = userData;
}
static std::string formatString(const char* fmt, ...)
{
	std::string s;
	va_list ap;
	va_start(ap, fmt);
	int size = vsnprintf(NULL, 0, fmt, ap);
	va_end(ap);
	if (size > 0) {
		s.resize(size);
		va_start(ap, fmt);
		// Writes the trailing '\0' as well, but we don't care.
		vsprintf(const_cast<char*>(s.data()), fmt, ap);
		va_end(ap);
	}
	return s;
}
void EasyCNN::easyAssertCore(const std::string& file, const std::string& function, const long line,
	const bool condition, const char* fmt, ...)
{
	if (!condition)
	{
		va_list args;
		va_start(args, fmt);
		const std::string errorStr = formatString(fmt, args);
		logFatal("FILE:%s,FUNCTION:%s,LINE:%d", file.c_str(), function.c_str(), line);
		logFatal(fmt, args);
		va_end(args);
		if (globalAssertFatalCB)
		{
			globalAssertFatalCB(globalAssertFatalUserData, errorStr);
		}
		else
		{
			assert(false);
			exit(0);
		}
	}
}
