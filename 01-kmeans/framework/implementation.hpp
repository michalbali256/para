#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <tbb/tbb.h> 

#include <vector>
#include <cmath>
#include <cstdint>

template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
public:

	static const std::vector<POINT> * points;
	static size_t k;
	std::vector<ASGN> * assignments;
	std::vector<POINT> * centroids;
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters)
	{
		/*
			Your core goes here ...
		 */
	}

	struct Sum {

		size_t k;
		std::vector<int_fast64_t> dist;
		std::vector<int_fast64_t> count;

		Sum() {}
		Sum(size_t k)
		{
			dist.resize(k);
			count.resize(k);
		}
		Sum(Sum& s, tbb::split)
		{
			dist.resize(k);
			count.resize(k);
		}

		void operator()(const tbb::blocked_range<POINT*>& r) {
			float temp = value;
			for (POINT* a = r.begin(); a != r.end(); ++a)
			{
				float m;

				for (size_t i = 0; i < k; ++i)
				{
					float dist = sqrt((a->x))
				}
			}
			value = temp;
		}
		void join(Sum& rhs) { value += rhs.value; }
	};


	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		Sum s;
		tbb::parallel_for(5, 6, [](int a) {});
		/*
			Your core goes here ...
		*/
		throw bpp::RuntimeError("Solution not implemented yet.");
	}
};


#endif
