#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <assert.h>
#include "process.hpp"
#include "packed_chunk.h"
#include "chunk_sender.h"
#include "chunk_receiver.h"

class process0 : public process
{
public:
	process0(int rank) : process(rank) {}

	int do_work(int argc, char** argv)
	{
		if (argc != 4)
		{
			std::cout << "Usage: matrix <input-A> <input-B> <output> \n";
			return 1;
		}

		std::ifstream fa(argv[1], std::ios::binary);
		std::ifstream fb(argv[2], std::ios::binary);


		if (!fa.good())
		{
			std::cout << "Could not open file " << argv[1] << "\n";
			return 3;
		}

		if (!fb.good())
		{
			std::cout << "Could not open file " << argv[2] << "\n";
			return 3;
		}


		uint32_t m, n, p, pb;
		fa.read((char*)& p, sizeof(float));
		fa.read((char*)& m, sizeof(float));
		fb.read((char*)& n, sizeof(float));
		fb.read((char*)& pb, sizeof(float));

		if (p != pb)
		{
			std::cout << "Cannot multiply matrices with sizes " << m << "x" << p << " and " << pb << "x" << n << ".\n";
			return 2;
		}

		//std::cout << "Sizes: " << m << "x" << p << " and " << pb << "x" << n << ".\n";

		/*                      +-------------------------------+
								|               n               |
								|                               |
								|                               |
							  p |             B                 |
								|                               |
								|                               |
								|                               |
				   p			+------------------------------+|
		+---------------------+
		|                     |
		|                     |
		| m        A          |
		|                     |
		|                     |
		+---------------------+
		*/

		ch_p = (p + (ch_size - 1)) / ch_size;
		//int ch_n = (n + (ch_size - 1)) / ch_size;
		//int ch_m = (m + (ch_size - 1)) / ch_size;

		int sizes[3] = {(int) m, (int) n, (int) p };
		

		MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);
		agree_comms();

		constexpr size_t mat_offset = 2 * sizeof(float);

		std::vector<float> result((size_t)m * n, 0);

		chunk_receiver<res_t> receiver(result.data(), m, n);
		
		int mom_rank = 1;
		
		size_t scatter_size = ch_p > worker_count_ ? worker_count_: ch_p;

		
		//std::this_thread::sleep_for(std::chrono::seconds(20));
		
		packed_chunk_array first(scatter_size+1);
		//packed_chunk_array second(scatter_size+1);

		packed_chunk_array* mom = &first;
		//packed_chunk_array* previous = &second;

		MPI_Request prev_req = MPI_REQUEST_NULL;
		
		float scatter_number = 0;

		for (size_t i_m = 0; i_m < m; i_m += ch_size)
		{
			for (size_t i_n = 0; i_n < n; i_n += ch_size)
			{
				range reduce_rng = get_range(mom_rank);

				size_t i_p = 0;
				
				while(i_p < p)
				{
					mom->fill_0();
					for (size_t w = 1; w <= scatter_size && i_p < p; ++w)
					{
						mom->range_begin(w) = (float)reduce_rng.begin;
						mom->range_end(w) = (float)reduce_rng.end;
						

						for (size_t i = i_m; i < i_m + ch_size && i < m; ++i)
						{
							size_t cp_size = i_p + ch_size > p ? p % ch_size : ch_size;
							fa.seekg((i * p + i_p) * sizeof(float) + mat_offset);
							fa.read((char*)mom->a_line(w, i - i_m), cp_size * sizeof(float));
						}

						for (size_t i = i_p; i < i_p + ch_size && i < p; ++i)
						{
							size_t cp_size = i_n + ch_size > n ? n % ch_size : ch_size;
							fb.seekg((i * n + i_n) * sizeof(float) + mat_offset);
							fb.read((char*)mom->b_line(w, i - i_p), cp_size * sizeof(float));
						}

						//std::cout << sender.chunks.size() << " " << i_m << " " << i_n << " " << i_p << " " << receiver.chunks.size() << " " << receiver.pending << "\n";

						next_rank(mom_rank);
						i_p += ch_size;
					}
					for (size_t w = 0; w <= scatter_size; ++w)
						mom->scatter_number(w) = scatter_number;

					/*if (prev_req != MPI_REQUEST_NULL)
					{
						MPI_Status status;
						MPI_Wait(&prev_req, &status);
					}*/

					//std::swap(mom,previous);
					/*std::cout << "ROOTscat " << reduce_rng.begin << " " << reduce_rng.end << " "
					 << previous.range_begin(0) << " " << previous.range_end(0) << " "
					 << previous.range_begin(1) << " " << previous.range_end(1) << " "
					  << previous.range_begin(2) << " " << previous.range_end(2) << "\n";
					for (size_t w = 0; w <= scatter_size; ++w)
					{
						std::cout << "P0A " << w << "\n";
						for (size_t z = 0; z < ch_size; ++z)
						{
							std::cout << "P0 ";
							for (size_t y = 0; y < ch_size; ++y)
								std::cout << mom->a(w)[z * ch_size + y] << " ";
							std::cout << "\n";
						}

						std::cout << "P0B " << w << "\n";
						for (size_t z = 0; z < ch_size; ++z)
						{
							std::cout << "P0 ";
							for (size_t y = 0; y < ch_size; ++y)
								std::cout << mom->b(w)[z * ch_size + y] << " ";
							std::cout << "\n";
						}
					}*/

					MPI_Scatter(mom->begin(), packed_chunk_size, MPI_FLOAT, MPI_IN_PLACE, packed_chunk_size, MPI_FLOAT, 0, comms[reduce_rng].comm);//, &prev_req);
					++scatter_number;
				}
				

				receiver.reduce_receive(comms[reduce_rng].comm, i_m, i_n);

			}
			//sender.wait_all_0();
			//receiver.wait_all();
		}
		
		fa.close();
		fb.close();
		
		
		receiver.wait_all();
		
		std::ofstream resf(argv[3], std::ios::binary);

		if (!resf.good())
		{
			std::cout << "Could not open file " << argv[3] << "\n";
			return 3;
		}

		/*m = 3;
		n = 12;
		result.resize(m * n);
		for (size_t i = 0; i < result.size(); ++i)
			result[i] = i;*/

		resf.write((const char*)& n, sizeof(int));
		resf.write((const char*)& m, sizeof(int));
		resf.write((const char*)result.data(), result.size() * sizeof(float));

		resf.close();

		return 0;
	}

};