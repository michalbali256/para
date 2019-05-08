#pragma once
#include <mpi.h>

#include <map>

#include "packed_chunk.h"



struct range
{
public:
	range() : begin(), end() {}
	range(int b, int e) : begin(b), end(e) {}
	int begin;
	int end;

	bool operator< (const range & rhs) const
	{
		if (begin == rhs.begin)
			return end < rhs.end;
		return begin < rhs.begin;
	}
};

struct comm_info
{
public:
	comm_info() : comm(), root() {}
	comm_info(MPI_Comm c, int r) : comm(c), root(r) {}
	MPI_Comm comm;
	int root;
};

class process
{
public:
	process(int rank): rank_(rank)
	{
		MPI_Comm_size(MPI_COMM_WORLD, &size_);
		worker_count_ = size_ - 1;
	}

	

	void agree_comms();

	range get_range(int rank);
protected:

	int rank_;
	int size_;
	int worker_count_;
	int ch_p;
	std::map<range, comm_info> comms;

	void next_rank(int & r)
	{
		++r;
		if (r == size_)
			r = 1;
	}

private:
	
};

range process::get_range(int mom_rank)
{
	range rng;
	if (ch_p >= worker_count_)
	{
		rng.begin = 1;
		rng.end = size_;
	}
	else
	{
		rng.begin = mom_rank;
		rng.end = mom_rank + ch_p;
		if (rng.end > size_)
		{
			rng.end = (rng.end % size_) + 1;
		}
	}
	return rng;
}

void process::agree_comms()
{
	if (ch_p >= worker_count_)
	{
		MPI_Comm c;
		MPI_Comm_dup(MPI_COMM_WORLD, &c);
		comms[{1, size_}] = { c, 0 };
		return;
	}
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	
	for (int i = 1; i < size_; ++i)
	{
		range rng = get_range(i);
		
		MPI_Group g;
		int ranges[2][3];

		if (rng.begin < rng.end)
		{
			//rank 0 is in every comm
			ranges[0][0] = 0; //first
			ranges[0][1] = 0; //last
			ranges[0][2] = 1; //stride

			ranges[1][0] = rng.begin; //first
			ranges[1][1] = rng.end - 1; //last
			ranges[1][2] = 1; //stride
		}
		else
		{
			ranges[0][0] = 0; //first
			ranges[0][1] = rng.end - 1; //last
			ranges[0][2] = 1; //stride

			ranges[1][0] = rng.begin; //first
			ranges[1][1] = size_ - 1; //last
			ranges[1][2] = 1; //stride
		}



		MPI_Group_range_incl(world_group, 2, ranges, &g);

		MPI_Comm newcomm;
		MPI_Comm_create(MPI_COMM_WORLD, g, &newcomm);
		if (newcomm != MPI_COMM_NULL)
		{
			/*int my_rank, s;
			MPI_Comm_rank(newcomm, &my_rank);
			MPI_Comm_size(newcomm, &s);
			if(rank_ == 0)
			std::cout << rank_ << " " << "[" << rng.begin << "," << rng.end << ") " << my_rank << " " << s << "\n";*/
			
			comms[rng] = { newcomm, 0 };
		}
	}
}