#ifndef SERIAL_POTENTIAL_IMPLEMENTATION_HPP
#define SERIAL_POTENTIAL_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <data.hpp>
#include "kernels.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>


/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
	std::vector<length_t> mLengths;	///< Reference to the graph lengths of the edges.
	std::vector<point_t> mVelocities;			///< Point velocity vectors.
					///< Preallocated buffer for force vectors.
	
	

	struct neigh_length
	{
		neigh_length() : neigh(), length() {}
		neigh_length(uint32_t neigh, uint32_t length) : neigh(neigh), length(length) {}
		uint32_t neigh;
		uint32_t length;
	};

	struct neigh_list
	{
		neigh_list(neigh_length * neigh, uint32_t count) : neigh(neigh), count(count) {}
		neigh_length * neigh = nullptr;
		uint32_t count;
	};

	std::vector<neigh_list> neighbors_;
	std::vector<neigh_length> edges_;
	std::vector<point_t> points1;
	std::vector<point_t> points2;
public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t>& lengths, index_t iterations)
	{
		/*
		 * Initialize your implementation.
		 * Allocate/initialize buffers, transfer initial data to GPU...
		 */
		
		mLengths = lengths;
		mVelocities.resize(points);
		

		std::vector<std::vector<neigh_length>> neigh_list;
		neigh_list.resize(points);

		for (size_t i = 0; i < edges.size(); ++i)
		{
			neigh_list[edges[i].p1].push_back({ edges[i].p2, lengths[i] });
			neigh_list[edges[i].p2].push_back({ edges[i].p1, lengths[i] });
		}
		edges_.resize(edges.size() * 2);

		size_t ind = 0;
		for (size_t i = 0; i < neigh_list.size(); ++i)
		{
			neighbors_.emplace_back(&edges_[ind], neigh_list[i].size());
			for (size_t j = 0; j < neigh_list[i].size(); ++j)
			{
				edges_[ind++] = neigh_list[i][j];
			}
		}

		first = true;
		points1.resize(points);
		points2.resize(points);

		/*CUCH(cudaSetDevice(0));
		CUCH(cudaMalloc((void**)& cu_data, count * sizeof(T)));
		CUCH(cudaMalloc((void**)& cu_res, count * sizeof(T)));
		bpp::Stopwatch cuda_swatch(true);

		CUCH(cudaMemcpy(cu_data, src.getRawData(), count * sizeof(T), cudaMemcpyHostToDevice));*/
	}

	real_t velocity_update_fact = this->mParams.timeQuantum / this->mParams.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
	bool first = true;
	point_t* in_points;
	point_t* out_points;

	virtual void iteration(std::vector<point_t>& points)
	{
		if (points.size() != mVelocities.size())
			throw (bpp::RuntimeError() << "Cannot compute next version of point positions."
				<< "Current model uses " << mVelocities.size() << " points, but the given buffer has " << points.size() << " points.");

		if (first)
		{
			first = false;
			points2 = points;
			in_points = points1.data();
			out_points = points2.data();
		}
		std::swap(out_points, in_points);
		
		
		for (index_t i = 0; i < points.size(); ++i)
		{
			point_t force(0.0,0.0);

			for (index_t j = 0; j < points.size(); ++j)
				addRepulsiveForce(in_points, i, j, force);

			for (size_t j = 0; j < neighbors_[i].count; ++j)
				addCompulsiveForce(in_points, i, neighbors_[i].neigh[j].neigh, neighbors_[i].neigh[j].length, force);
			
			double vel_x = (mVelocities[i].x + (real_t)force.x * velocity_update_fact) * this->mParams.slowdown;
			double vel_y = (mVelocities[i].y + (real_t)force.y * velocity_update_fact) * this->mParams.slowdown;
			mVelocities[i].x = vel_x;
			mVelocities[i].y = vel_y;
				
			out_points[i].x = in_points[i].x + vel_x * this->mParams.timeQuantum;
			out_points[i].y = in_points[i].y + vel_y * this->mParams.timeQuantum;
		}

		if (out_points == points1.data())
			points = points1;
		else
			points = points2;
		
	}


	virtual void getVelocities(std::vector<point_t>& velocities)
	{
		/*
		 * Retrieve the velocities buffer.
		 * This operation is for vreification only and it does not have to be efficient.
		 */
		velocities = mVelocities;
	}
private:
	
	void addRepulsiveForce(const point_t * points, index_t p1, index_t p2, point_t & force)
	{
		real_t dx = (real_t)points[p1].x - (real_t)points[p2].x;
		real_t dy = (real_t)points[p1].y - (real_t)points[p2].y;
		real_t sqLen = std::max<real_t>(dx * dx + dy * dy, (real_t)0.0001);
		real_t fact = this->mParams.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));	// mul factor
		dx *= fact;
		dy *= fact;
		force.x += dx;
		force.y += dy;
	}


	
	void addCompulsiveForce(const point_t * points, index_t p1, index_t p2, length_t length, point_t & force)
	{
		real_t dx = (real_t)points[p2].x - (real_t)points[p1].x;
		real_t dy = (real_t)points[p2].y - (real_t)points[p1].y;
		real_t sqLen = dx * dx + dy * dy;
		real_t fact = (real_t)std::sqrt(sqLen) * this->mParams.edgeCompulsion / (real_t)(length);
		dx *= fact;
		dy *= fact;
		force.x += dx;
		force.y += dy;
	}


	
	void updateVelocities(const std::vector<point_t>& forces)
	{
		real_t fact = this->mParams.timeQuantum / this->mParams.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
		for (std::size_t i = 0; i < mVelocities.size(); ++i) {
			mVelocities[i].x = (mVelocities[i].x + (real_t)forces[i].x * fact) * this->mParams.slowdown;
			mVelocities[i].y = (mVelocities[i].y + (real_t)forces[i].y * fact) * this->mParams.slowdown;
		}
	}

	void computeForces(std::vector<point_t>& points, std::vector<point_t>& forces)
	{
		forces.resize(points.size());

		// Compute repulsive forces between all vertices.
		for (index_t i = 0; i < forces.size(); ++i)
		{
			forces[i].x = forces[i].y = (real_t)0.0;

			for (index_t j = 0; j < forces.size(); ++j)
				addRepulsiveForce(points, i, j, forces);
		
			for (size_t j = 0; j < neighbors_[i].count; ++j)
			{
				addCompulsiveForce(points, i, neighbors_[i].neigh[j].neigh, neighbors_[i].neigh[j].length, forces);
			}
		}
	}
};


#endif