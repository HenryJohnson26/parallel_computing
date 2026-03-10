#include "make_unique.h"
#include "world.h"
#include "quad-tree.h"
#include <algorithm>
#include <iostream>

// TASK 2

// NOTE: You may modify this class definition as you see fit, as long as the class name,
// and type of simulateStep and buildAccelerationStructure remain the same.

const int QuadTreeLeafSize = 8;
const int QuadTreeParallelThreshold = 1024;
class ParallelNBodySimulator : public INBodySimulator
{
public:
    // TODO: implement a function that builds and returns a quadtree containing particles.
    // You do not have to preserve this function type.
    std::shared_ptr<QuadTreeNode> buildQuadTree(std::vector<Particle> & particles, Vec2 bmin, Vec2 bmax, int depth)
    {
        if (particles.size()<QuadTreeLeafSize){
            // return leaf node
            auto leaf = std::make_shared<QuadTreeNode>();
            leaf->isLeaf = true;
            leaf->particles = particles;
            return leaf;
        }
        else{
            auto nonleaf = std::make_shared<QuadTreeNode>();
            Vec2 pivot = (bmin + bmax) * 0.5f;
            std::vector<Particle> q0, q1, q2, q3;
            for (auto & p : particles) {
                bool right = p.position.x >= pivot.x;
                bool below = p.position.y >= pivot.y;
                if (!right && !below) q0.push_back(p); // top-left
                else if ( right && !below) q1.push_back(p); // top-right
                else if (!right &&  below) q2.push_back(p); // bottom-left
                else q3.push_back(p); // bottom-right
            }

            // TODO: check that depth isn't too far down
           // Only spawn tasks if there's enough work to justify the overhead
            if (particles.size() >= QuadTreeParallelThreshold) {
                #pragma omp task shared(nonleaf) firstprivate(q0, bmin, pivot, depth)
                nonleaf->children[0] = buildQuadTree(q0, bmin, pivot, depth + 1);

                #pragma omp task shared(nonleaf) firstprivate(q1, bmin, bmax, pivot, depth)
                nonleaf->children[1] = buildQuadTree(q1, Vec2(pivot.x, bmin.y), Vec2(bmax.x, pivot.y), depth + 1);

                #pragma omp task shared(nonleaf) firstprivate(q2, bmin, bmax, pivot, depth)
                nonleaf->children[2] = buildQuadTree(q2, Vec2(bmin.x, pivot.y), Vec2(pivot.x, bmax.y), depth + 1);

                #pragma omp task shared(nonleaf) firstprivate(q3, bmin, bmax, pivot, depth)
                nonleaf->children[3] = buildQuadTree(q3, pivot, bmax, depth + 1);

                #pragma omp taskwait
            } else {
                nonleaf->children[0] = buildQuadTree(q0, bmin, pivot, depth + 1);
                nonleaf->children[1] = buildQuadTree(q1, Vec2(pivot.x, bmin.y), Vec2(bmax.x, pivot.y), depth + 1);
                nonleaf->children[2] = buildQuadTree(q2, Vec2(bmin.x, pivot.y), Vec2(pivot.x, bmax.y), depth + 1);
                nonleaf->children[3] = buildQuadTree(q3, pivot, bmax, depth + 1);
            }


            return nonleaf;
       }
    }

    // Do not modify this function type.
    virtual std::shared_ptr<AccelerationStructure> buildAccelerationStructure(std::vector<Particle> & particles)
    {
        // build quad-tree
        auto quadTree = std::make_shared<QuadTree>();

        // find bounds
        Vec2 bmin(1e30f, 1e30f);
        Vec2 bmax(-1e30f, -1e30f);

        for (auto & p : particles)
        {
            bmin.x = fminf(bmin.x, p.position.x);
            bmin.y = fminf(bmin.y, p.position.y);
            bmax.x = fmaxf(bmax.x, p.position.x);
            bmax.y = fmaxf(bmax.y, p.position.y);
        }

        quadTree->bmin = bmin;
        quadTree->bmax = bmax;

        #pragma omp parallel
        #pragma omp single
        // build nodes
        quadTree->root = buildQuadTree(particles, bmin, bmax, 0);
        if (!quadTree->checkTree()) {
          std::cout << "Your Tree has Error!" << std::endl;
        }

        return quadTree;
    }

    // Do not modify this function type.
    virtual void simulateStep(AccelerationStructure * accel,
                            std::vector<Particle> & particles,
                            std::vector<Particle> & newParticles,
                            StepParameters params) override
    {
        // TODO: implement parallel version of quad-tree accelerated n-body simulation here,
        // using quadTree as acceleration structure
        auto qtree = static_cast<QuadTree*>(accel);
        int n = particles.size();

        #pragma omp parallel for schedule(dynamic, n/128)
        for(int i = 0; i < (int)particles.size(); i++){
            auto p = particles[i];
            std::vector<Particle> local_ps;
            qtree->getParticles(local_ps, p.position, params.cullRadius);
            int n = local_ps.size();
            {
                Vec2 force(0.0f, 0.0f);
                for (auto & other : local_ps){
                    if(other.id != p.id){
                        force = force + computeForce(p, other, params.cullRadius);
                    }
                }
                newParticles[i] = updateParticle(p, force, params.deltaTime);
            }
        }
    }
};

// Do not modify this function type.
std::unique_ptr<INBodySimulator> createParallelNBodySimulator()
{
  return std::make_unique<ParallelNBodySimulator>();
}