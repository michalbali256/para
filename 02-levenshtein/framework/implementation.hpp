#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>

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
		DIST h_size = len1;
		DIST v_size = len2;

		if (h_size < v_size)
			std::swap(v_size, h_size);

		carry_h.resize(h_size);
		carry_v.resize(v_size);



		ch_h_count = h_size / ch_h_size;
		ch_v_count = v_size / ch_v_size;

		if (h_size % ch_h_size != 0)
			std::cout << "Not divisible horisontal\n";
		if (v_size % ch_v_size != 0)
			std::cout << "Not divisible vertical\n";

		deps_global = new DIST *[ch_h_count+1];
		for (size_t i = 0; i < ch_h_count+1; ++i)
			deps_global[i] = new DIST[ch_v_count+1];
	}

	DIST * * deps_global;

	DIST ch_v_count;
	DIST ch_h_count;

	DIST ch_v_size = 256;
	DIST ch_h_size = 256;

	DIST v_size;
	DIST h_size;

	std::vector<DIST> carry_h;
	std::vector<DIST> carry_v;

	const C* horis;
	const C* vert;

	DIST compute_one(DIST left, DIST upper, DIST upperleft, size_t i_v, size_t i_h)
	{
		return std::min<DIST>({ left + 1, upper + 1, upperleft + (vert[i_v] == horis[i_h] ? 0 : 1) });
	}

	void compute_chunk(size_t ch_h, size_t ch_v)
	{

		for (size_t i = ch_v * ch_v_size; i < (ch_v + 1) * ch_v_size; ++i)
		{
			DIST ul, l;
			ul = deps_global[ch_h][ch_v];

			l = carry_v[i];

			for (size_t j = ch_h * ch_h_size; j < (ch_h + 1) * ch_h_size; ++j)
			{
				DIST u = carry_h[j];
				DIST n = compute_one(l, u, ul, i, j);
				carry_h[j] = n;
				l = n;
				ul = u;
			}

			
			carry_v[i] = carry_h[(ch_h + 1) * ch_h_size - 1];
		}

		deps_global[ch_h+1][ch_v+1] = carry_h[(ch_h + 1) * ch_h_size - 1];
	}

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		DIST * * deps = deps_global;

		for (size_t i = 0; i < carry_h.size(); ++i)
		{
			carry_h[i] = i + 1;
		}
		
		for (size_t i = 0; i < carry_v.size(); ++i)
		{
			carry_v[i] = i + 1;
		}

		for (size_t i = 0; i < ch_h_count; ++i)
		{
			deps[i][0] = i * ch_h_size;
		}

		for (size_t i = 0; i < ch_v_count; ++i)
		{
			deps[0][1] = i * ch_v_size;
		}

		// Special case (one of the strings is empty).
		//f (mRow.size() == 0)
		//	return rows;

		// Make sure s1 is the longer string and s2 is the longer one.
		horis = &str1[0];
		vert = &str2[0];
		if (str1.size() < str2.size())
			std::swap(horis, vert);

		

		#pragma omp parallel shared(carry_h, carry_v)
		{
			#pragma omp single
			{
				#pragma omp task depend( out: deps[0][0] )
				{
					compute_chunk(0, 0);
				}
				//std::cout << "first\n";

				for (size_t i = 1; i < ch_h_count; ++i)
				{
					#pragma omp task depend( out: deps[i][0]) depend(in : deps[i-1][0])
					{
						compute_chunk(i, 0);
					}
					//std::cout << "second\n";
				}

				for (size_t j = 1; j < ch_v_count; ++j)
				{
					#pragma omp task depend( out: deps[0][j]) depend(in : deps[0][j-1])
					{
						compute_chunk(0, j);
					}
				}

				for (size_t i = 1; i < ch_h_count; ++i)
				{
					for (size_t j = 1; j < ch_v_count; ++j)
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

		return carry_h[carry_h.size()-1];
	}

	~EditDistance()
	{
		
		for (size_t i = 0; i < ch_h_count+1; ++i)
			delete[] deps_global[i];
		delete[] deps_global;
	}

};


#endif
