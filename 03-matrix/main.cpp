#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "process0.hpp"
#include "processx.hpp"


int main(int argc, char * * argv)
{
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int ret;
	if (rank == 0)
	{
		process0 p(rank);
		ret = p.do_work(argc, argv);
	}
	else
	{
		processx p(rank);
		ret = p.do_work();
	}
	

	MPI_Finalize();
	return ret;
}