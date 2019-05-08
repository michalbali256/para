#pragma once
#include <deque>

#include <mpi.h>

#include "packed_chunk.h"


template <typename T>
class chunk_sender
{
public:
	class chunk_info
	{
	public:
		chunk_info() : ch(), req() {}
		T ch;
		MPI_Request req;
	};

	std::deque<chunk_info> chunks;

	T & create()
	{
		chunks.emplace_back();
		//check_done();
		return chunks.back().ch;
	}

	void send_back(int dest);
	void send_back_tag(int dest, int tag);
	void reduce_back(int dest, MPI_Comm comm);

	void check_done()
	{
		if (chunks.empty())
			return;
		int flag = 1;
		while (flag && chunks.size() > 1)
		{
			MPI_Status status;
			MPI_Test(&chunks.front().req, &flag, &status);
			if (flag)
			{
				chunks.pop_front();
			}
		} 
	}

	void wait_all()
	{
		while (chunks.size() > 1)
		{
			MPI_Status status;
			MPI_Wait(&chunks.front().req, &status);
			chunks.pop_front();
		}
	}
	void wait_all_0()
	{
		while (chunks.size() > 0)
		{
			MPI_Status status;
			MPI_Wait(&chunks.front().req, &status);
			chunks.pop_front();
		}
	}

};



template<>
void inline chunk_sender<packed_chunk>::send_back_tag(int dest, int tag)
{
	MPI_Issend(chunks.back().ch.begin(), packed_chunk_size, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, &chunks.back().req);
}

template<>
void inline chunk_sender<packed_chunk>::send_back(int dest)
{
	send_back_tag(dest, 0);
}

template<>
chunk_sender<res_t>::chunk_info::chunk_info() : ch(), req()
{
	/*for (int i = 0; i < ch_area; ++i)
		ch[i] = 0;*/
}

template<>
void chunk_sender<res_t>::reduce_back(int dest, MPI_Comm comm)
{

	MPI_Reduce(chunks.back().ch.data(), NULL, ch_area, MPI_FLOAT, MPI_SUM, 0, comm);// , & chunks.back().req);
	chunks.pop_back();
}
