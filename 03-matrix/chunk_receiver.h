#pragma once
#pragma once
#include <deque>

#include <mpi.h>

#include <cstring>

#include "packed_chunk.h"


template <typename T>
class chunk_receiver
{
	float* result_;

	
	size_t m_, n_;
	T zeros_;
public:
	chunk_receiver(float* result, size_t m, size_t n) : result_(result), m_(m), n_(n)
	{
		init_zeros();
	}

	size_t pending = 0;

	class chunk_info
	{
	public:
		chunk_info(size_t im, size_t in) : ch(), req(), i_m(im), i_n(in) {}
		T ch;
		MPI_Request req;
		size_t i_m;
		size_t i_n;
	};
//-7 15 7 -15 -45 -11 -45 -9 -19 -17 15 -3 -35 -13 17 -21 17 17 11 -7 -29 1 1 -13 -47
//7 -13 17 -7 -11 11 -21 -43 1 -31 -35 -23 -1 -9 15 -31 -7 11 43 11 37 -19 27 -29 31
	std::deque<chunk_info> chunks;

	void init_zeros();

	void reduce_receive(MPI_Comm comm, size_t i_m, size_t i_n)
	{
		check_done();
		chunks.emplace_back(i_m, i_n);

		MPI_Ireduce(zeros_.data(), chunks.back().ch.data(), ch_area, MPI_FLOAT, MPI_SUM, 0, comm, & chunks.back().req);

		/*for (size_t z = 0; z < ch_size; ++z)
		{
			std::cout << "RES ";
			for (size_t y = 0; y < ch_size; ++y)
				std::cout << chunks.back().ch.data()[z * ch_size + y] << " ";
			std::cout << "\n";
		}*/
	}

	void process_one()
	{
		size_t i_m = chunks[pending].i_m;
		size_t i_n = chunks[pending].i_n;

		size_t cp_size = i_n + ch_size >= n_ ? n_ % ch_size : ch_size;
		for (size_t i = i_m; i < i_m + ch_size && i < m_; ++i)
		{
			memcpy(&result_[i * n_ + i_n], &chunks[pending].ch.data()[(i - i_m) * ch_size], cp_size * sizeof(float));
		}

		++pending;
	}

	void check_done()
	{		
		if (chunks.empty())
			return;
		int flag = 1;
		while (flag && pending < chunks.size())
		{
			MPI_Status status;
			MPI_Test(&chunks[pending].req, &flag, &status);
			if (flag)
			{
				process_one();
			}
		}
	}

	void wait_all()
	{
		if (chunks.empty())
			return;
		while (pending < chunks.size())
		{
			MPI_Status status;
			MPI_Wait(&chunks[pending].req, &status);
			process_one();
		}
	}
};

template<>
void chunk_receiver<res_t>::init_zeros()
{
	//for (size_t i = 0; i < ch_area; ++i)
	//	zeros_[i] = 0;
}