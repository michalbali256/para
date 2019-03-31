#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		ch_h_count = 5;
		ch_v_count = 4;
		
		deps_global = new char *[ch_h_count];
		for(size_t i = 0; i < ch_h_count; ++i)
			deps_global[i] = new char[ch_v_count];
	}
	
	char * * deps_global;
	
	size_t ch_v_count;
	size_t ch_h_count;
	
	void compute_chunk(size_t ch_i, size_t ch_j)
	{
		//std::cout << ch_i << "starting" << ch_j << "\n";
		size_t per = 5 - ch_i;
		std::this_thread::sleep_for(2s);
		std::cout << ch_i << " " << ch_j << "\n";
		std::cout.flush();
	}
	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		// Number of rows (size of the larger of the two strings).
		std::size_t rows = std::min<std::size_t>(str1.size(), str2.size());

		// Special case (one of the strings is empty).
		//f (mRow.size() == 0)
		//	return rows;

		// Make sure s1 is the longer string and s2 is the longer one.
		const C* horis = &str1[0];
		const C* vert = &str2[0];
		if (str1.size() < str2.size())
			std::swap(horis, vert);
		
		char * * deps = deps_global; 
		
		#pragma omp parallel
		{
			#pragma omp single
			{
			#pragma omp task depend( out: deps[0][0] )
			{
				compute_chunk(0,0);
			}
			std::cout << "first\n";
			
			for(size_t i = 1; i < ch_h_count; ++i)
			{
				#pragma omp task depend( out: deps[i][0]) depend(in : deps[i-1][0])
				{
					compute_chunk(i, 0);
				}
				std::cout << "second\n";
			}
			
			for(size_t j = 1; j < ch_v_count; ++j)
			{
				#pragma omp task depend( out: deps[0][j]) depend(in : deps[0][j-1])
				{
					compute_chunk(0, j);
				}
			}
			
			for(size_t i = 1; i < ch_h_count; ++i)
			{
				for(size_t j = 1; j < ch_v_count; ++j)
				{
					#pragma omp task depend( out: deps[i][j]) depend(in : deps[i-1][j]) depend(in : deps[i][j-1])
					{
						compute_chunk(i, j);
					}
				}
			}
			
			
			}
			#pragma omp taskwait
		}
		
		return 0;
	}
	
	
	
};


#endif
