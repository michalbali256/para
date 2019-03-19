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
	static const std::vector<POINT> * points;
	static size_t k;
	static std::vector<ASGN> assignments;
	static std::vector<POINT> * centroids;

public:

	
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters)
	{
		assignments.resize(points);
		/*
			Your core goes here ...
		 */
	}

	template <bool update_assignment>
	struct sum {
		std::vector<POINT> point_sum;
		std::vector<int_fast64_t> centroid_count;

		sum()
		{
			point_sum.resize(k);
			centroid_count.resize(k);
		}
		sum(sum& s, tbb::split)
		{
			
			point_sum.resize(k);
			centroid_count.resize(k);
		}

		typedef typename POINT::coord_t coord_t;

		static coord_t distance(const POINT &point, const POINT &centroid)
		{
			std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
			std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
			return (coord_t)(dx*dx + dy * dy);
		}

		static std::size_t getNearestCluster(const POINT &point, const std::vector<POINT> &centroids)
		{
			coord_t minDist = distance(point, centroids[0]);
			std::size_t nearest = 0;
			for (std::size_t i = 1; i < centroids.size(); ++i)
			{
				coord_t dist = distance(point, centroids[i]);
				if (dist < minDist) {
					minDist = dist;
					nearest = i;
				}
			}

			return nearest;
		}

		void operator()(const tbb::blocked_range<const POINT*>& r)
		{
			
			for (const POINT* a = r.begin(); a != r.end(); ++a)
			{
				size_t mi = getNearestCluster(*a, *centroids);

				point_sum[mi].x += a->x;
				point_sum[mi].y += a->y;
				centroid_count[mi]++;

				if(update_assignment)
					assignments[a - points->data()] = (ASGN) mi;
			}
			
		}
		void join(sum& rhs)
		{
			for (size_t i = 0; i < k; ++i)
			{
				point_sum[i].x += rhs.point_sum[i].x;
				point_sum[i].y += rhs.point_sum[i].y;
				centroid_count[i] += rhs.centroid_count[i];
			}
		}
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
	template <bool update_assignment>
	void do_one_iter(const std::vector<POINT> &points, std::vector<POINT> &centroids)
	{
		sum<update_assignment> s;
		tbb::blocked_range<const POINT *> whole(points.data(), points.data() + points.size());

		tbb::parallel_reduce(whole, s);

		for (std::size_t i = 0; i < k; ++i)
		{
			if (s.centroid_count[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
			centroids[i].x = s.point_sum[i].x / (std::int64_t)s.centroid_count[i];
			centroids[i].y = s.point_sum[i].y / (std::int64_t)s.centroid_count[i];
		}
	}

	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		
		
		centroids.resize(k);
		for (std::size_t i = 0; i < k; ++i)
			centroids[i] = points[i];

		KMeans::points = &points;
		KMeans::centroids = &centroids;
		KMeans::k = k;

		for (size_t i = 0; i < iters-1; ++i)
		{
			do_one_iter<false>(points, centroids);
		}
		do_one_iter<true>(points, centroids);
		assignments = std::move(KMeans::assignments);

	}
};

template<typename POINT, typename ASGN, bool DEBUG>
size_t KMeans<POINT, ASGN, DEBUG>::k;

template<typename POINT, typename ASGN, bool DEBUG>
const std::vector<POINT> * KMeans<POINT, ASGN, DEBUG>::points;

template<typename POINT, typename ASGN, bool DEBUG>
std::vector<ASGN> KMeans<POINT, ASGN, DEBUG>::assignments;

template<typename POINT, typename ASGN, bool DEBUG>
std::vector<POINT> * KMeans<POINT, ASGN, DEBUG>::centroids;

#endif
