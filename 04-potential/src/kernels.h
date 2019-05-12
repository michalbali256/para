#ifndef CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H
#define CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H

//#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cstdint>

#ifndef _MSC_VER

#include <cuda_runtime.h>

using cudaError_t = int;
#define cudaSuccess 0
/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception
{
	
	
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	virtual ~CudaError() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = NULL)
{
	if (status != cudaSuccess) {
		throw (CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)
#else
#define CUCH(status) status
#endif

/**
 * Macro wrapper for CUDA calls checking.
 */



struct neigh_length
{
	neigh_length() : neigh(), length() {}
	neigh_length(uint32_t neigh, uint32_t length) : neigh(neigh), length(length) {}
	uint32_t neigh;
	uint32_t length;
};

struct neigh_list
{
	neigh_list(neigh_length* neigh, uint32_t count) : neigh(neigh), count(count) {}
	neigh_length* neigh = nullptr;
	uint32_t count;
};



/*
 * Kernel wrapper declarations.
 */

void run_my_kernel(float *src);



#endif
