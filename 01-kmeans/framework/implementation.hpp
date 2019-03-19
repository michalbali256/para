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
	static std::vector<ASGN> * assignments;
	static std::vector<POINT> * centroids;
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

	struct sum {

		size_t k;
		std::vector<POINT> point_sum;
		std::vector<int_fast64_t> count;

		sum() {}
		sum(size_t k)
		{
			point_sum.resize(k);
			count.resize(k);
		}
		sum(sum& s, tbb::split)
		{
			point_sum.resize(k);
			count.resize(k);
		}

		void operator()(const tbb::blocked_range<POINT*>& r) {
			
			for (POINT* a = r.begin(); a != r.end(); ++a)
			{
				float m;
				size_t mi;
				for (size_t i = 0; i < k; ++i)
				{
					float dist = sqrt((a->x - (*centroids)[i].x) * (a->x - (*centroids)[i].x) + (a->y - (*centroids)[i].y) * (a->y - (*centroids)[i].y));
					if (dist < m)
					{
						mi = i;
						m = dist;
					}
				}
				point_sum[mi].x += a->x;
				point_sum[mi].y += a->y;

			}
			
		}
		void join(sum& rhs) { }
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
		sum s;
		tbb::parallel_for(5, 6, [](int a) {});
		/*
			Your core goes here ...
		*/
		throw bpp::RuntimeError("Solution not implemented yet.");
	}
};


#endif
