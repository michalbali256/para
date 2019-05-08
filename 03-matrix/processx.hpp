#pragma once

#include <memory>
#include <assert.h>
#include <mpi.h>

#include "process.hpp"

#ifdef _MSC_VER
	#define __restrict__ __restrict
#endif

#ifndef __restrict__ 
#define __restrict__ 
#endif

class processx : public process
{
public:
	processx(int rank) : process(rank) {}

	int do_work();

	void do_chunk(float * __restrict__ a, float* __restrict__ b, float* __restrict__ dst);
	

};



int processx::do_work()
{
	int sizes[3];
	MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);
	size_t m = sizes[0];
	size_t n = sizes[1];
	size_t p = sizes[2];


	ch_p = (p + (ch_size - 1)) / ch_size;

	agree_comms();
	//std::this_thread::sleep_for(std::chrono::seconds(20));
	int mom_rank = 1;

	
	res_t sum;

	size_t scatter_size = ch_p > worker_count_ ? worker_count_ : ch_p;

	float scatter_number = 0;
	//std::cout << "scatter_size " << scatter_size << "\n";
	for (size_t i_m = 0; i_m < m; i_m += ch_size)
	{
		for (size_t i_n = 0; i_n < n; i_n += ch_size)
		{
			range reduce_rng = get_range(mom_rank);

			size_t i_p = 0;
			bool did_work = false;

			sum.fill_0();

			while (i_p < p)
			{
				if (ch_p > worker_count_)
					mom_rank = 1;
				bool this_turn = false;
				for (size_t w = 1; w <= scatter_size && i_p < p; ++w)
				{
					if (mom_rank == rank_)
					{
						//std::cout << rank_ << " " << i_m << " " << i_n << " " << i_p << "\n"; 
						did_work = true;
						this_turn = true;
					}
					next_rank(mom_rank);
					i_p += ch_size;
				}
				if (!did_work && ch_p < worker_count_)
					continue;

				packed_chunk ch;
				
				MPI_Scatter(nullptr, 0, MPI_FLOAT, ch.begin(), packed_chunk_size, MPI_FLOAT, 0, comms[reduce_rng].comm);//, &req);

				
				/*for (size_t z = 0; z < ch_size; ++z)
				{
					std::cout << "W" << rank_ << " ";
					for (size_t y = 0; y < ch_size; ++y)
						std::cout << ch.a()[z * ch_size + y] << " ";
					std::cout << "\n";
				}

				for (size_t z = 0; z < ch_size; ++z)
				{
					std::cout << "W" << rank_ << " ";
					for (size_t y = 0; y < ch_size; ++y)
						std::cout << ch.b()[z * ch_size + y] << " ";
					std::cout << "\n";
				}*/

				/*if (ch.scatter_number() != scatter_number)
					std::cout << "NOTEQUAL " << ch.scatter_number() << " " << scatter_number << "\n";
				++scatter_number;*/
				if (ch.range_begin() != reduce_rng.begin && this_turn)
				{
					std::cout << "first " << rank_ << " " << ch.range_begin() << " " << reduce_rng.begin << " " << i_m << " " << i_n << " " << i_p << "\n";
				}
				if (ch.range_end() != reduce_rng.end && this_turn)
				{
					std::cout << "end " << rank_ << " " << ch.range_end() << " " << reduce_rng.end << " " << i_m << " " << i_n << " " << i_p << "\n";
				}

				do_chunk(ch.a(), ch.b(), sum.data());
				/*for (size_t z = 0; z < ch_size; ++z)
				{
					std::cout << "W" << rank_ << " ";
					for (size_t y = 0; y < ch_size; ++y)
						std::cout << sum.data()[z * ch_size + y] << " ";
					std::cout << "\n";
				}*/
			}

			/*if (i_n == 0 && i_m == 0)
			{
				for (size_t i = 0; i < 10; ++i)
					std::cout << sum.data()[i] << " ";
				std::cout << "\n";
			}*/

			if (did_work || ch_p >= worker_count_)
			{
				MPI_Request req;
				MPI_Ireduce(sum.data(), NULL, ch_area, MPI_FLOAT, MPI_SUM, 0, comms[reduce_rng].comm, &req);
				MPI_Status s;
				MPI_Wait(&req, &s);
			}
		}
	}

	return 0;
}


void processx::do_chunk(float* a, float* b, float* dst)
{
	for (size_t i = 0; i < ch_size; ++i)
	{
		for (size_t j = 0; j < ch_size; ++j)
		{
			float sum = 0;
			for (size_t k = 0; k < ch_size; ++k)
			{
				sum += a[i * ch_size + k] * b[k * ch_size + j];
			}
			dst[i * ch_size + j] += sum;
		}
	}
}