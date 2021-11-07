module DirectQHull

# MIT License
# 
# Copyright (c) 2021 Juha Tapio Heiskala
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# function "qh_get_extremes_2d" defined later licensed under BSD license from Scipy

import Qhull_jll

import Base.getproperty
import Base.iterate
import Base.getindex

# define Qhull types

QHboolT = Cuint
QHrealT = Cdouble
QHcoordT = QHrealT
QHpointT = QHcoordT
QHintT = Cint 
QHprintT = Cint
QHcharT = Cchar
QHuintT = Cuint
QHulongT = Culong
QHcenterT = Cint
QHdoubleT = Cdouble
QHvoidT = Cvoid
QHfileT = Cvoid


mutable struct qhT
end

# defines clockwise or counterclockwise orientation e.g. 2d-convex hull vertex ordering
const qh_ORIENTclock = 0

const qh_lib = Qhull_jll.get_libqhull_r_path()

function qh_alloc_qh(err_file::Ptr{Cvoid}=C_NULL)
    qh_ptr = ccall((:qh_alloc_qh, qh_lib), Ptr{qhT}, (Ptr{Cvoid},), err_file)    
    (qh_ptr != C_NULL) ? qh_ptr : throw(ErrorException("qhT initialization failure."))
end

# This should always be true
@assert(sizeof(Ptr{Cvoid}) == sizeof(Int))

abstract type QHsetelemT end

# this is not an exact representation of QHull's set type
# Qhull set is dynamically allocated so exact representation is not straighforward as Julia type.
# (would require NTuple type for elements with tuple length equal to number elements in the list)
# QH set element type is defined as union of void pointer and integer
struct QHsetT{T<:Union{QHintT, Ptr{<:QHsetelemT}}}
    maxsize::QHintT          # /* maximum number of elements (except NULL) */
    e::Array{T}              # /* array of pointers, tail is NULL */
                             # /* last slot (unless NULL) is actual size+1
                             # /* e[maxsize]==NULL or e[e[maxsize]-1]==NULL */
                             # /* this may generate a warning since e[] contains  maxsize elements */
    function QHsetT{T}(ptr::Ptr{QHsetT{T}}) where T<:Ptr{<:QHsetelemT}
        max_size = unsafe_load(Ptr{QHintT}(ptr))
        # with passing C_NULL as qh_ptr below the call will crash, if the setsize is invalid
        # however, qhull would exit in this case with internal error in a more gracious manner
        set_size = qh_setsize(Ptr{qhT}(C_NULL), ptr)
        # assumed here QHsetT field e is offset Ptr-size from the maxsize field
        ptr_array = unsafe_wrap(Array, Ptr{T}(ptr+sizeof(Ptr)), set_size)
        new(max_size, ptr_array)
    end
end

function Base.getindex(set::QHsetT{T}, idx::Int) where T<:Union{QHintT, Ptr{<:QHsetelemT}}
    ptr = set.e[idx]
    unsafe_load(ptr)
end


# Qhull vertex type
mutable struct QHvertexT{HD} <: QHsetelemT
    next::Ptr{QHvertexT{HD}}                # /* next vertex in vertex_list or vertex_tail */
    previous::Ptr{QHvertexT{HD}}            # /* previous vertex in vertex_list or NULL, for C++ interface */
    point::Ptr{NTuple{HD, QHpointT}}        # /* hull_dim coordinates (coordT) */
    neighbors::Ptr{QHsetT{<:QHsetelemT}}    # /* neighboring facets of vertex, qh_vertexneighbors()
                                            # initialized in io_r.c or after first merge
                              # qh_update_vertices for qh_addpoint or qh_triangulate
                              # updated by merges
                              # qh_order_vertexneighbors by 2-d (orientation) 3-d (adjacency), n-d (f.visitid,id) */
    id::QHuintT               # /* unique identifier, 1..qh.vertex_id,  0 for sentinel, printed as 'r%d' */
    visitid::QHuintT          # /* for use with qh.vertex_visit, size must match */
    flags::QHcharT
    # seen:1;      /* used to perform operations only once */
    #flagT    seen2:1;     /* another seen flag */
    #flagT    deleted:1;   /* vertex will be deleted via qh.del_vertices */
    #flagT    delridge:1;  /* vertex belonged to a deleted ridge, cleared by qh_reducevertices */
    #flagT    newfacet:1;  /* true if vertex is in a new facet
    #                       vertex is on qh.newvertex_list and it has a facet on qh.newfacet_list
    #                       or vertex is on qh.newvertex_list due to qh_newvertices while merging
    #                       cleared by qh_resetlists */
    #flagT    partitioned:1; /* true if deleted vertex has been partitioned */
end

# Qhull facet type
mutable struct QHfacetT{HD} <: QHsetelemT
    furthestdist::QHcoordT  # distance to furthest point of outsideset
    maxoutside::QHcoordT    # max computed distance of point to facet
                            # Before QHULLfinished this is an approximation
                            # since maxdist not always set for qh_mergefacet
                            # Actual outer plane is +DISTround and
                            # computed outer plane is +2*DISTround.
                            # Initial maxoutside is qh.DISTround, otherwise distance tests need to account for DISTround */

    offset::QHcoordT        # exact offset of hyperplane from origin 
    normal::Ptr{NTuple{HD, QHcoordT}}   # normal of hyperplane, hull_dim coefficients 
                            # if f.tricoplanar, shared with a neighbor

    #union {                # in order of testing */
    area::QHrealT           # area of facet, only in io_r.c if  f.isarea */
    #facetT *replace    # replacement facet for qh.NEWfacets with f.visible
                            # NULL if qh_mergedegen_redundant, interior, or !NEWfacets */
                            # facetT *samecycle;   /* cycle of facets from the same visible/horizon intersection,
                            # if ->newfacet */
    # facetT *newcycle;    /*  in horizon facet, current samecycle of new facets */
    # facetT *trivisible;  /* visible facet for ->tricoplanar facets during qh_triangulate() */
    # facetT *triowner;    /* owner facet for ->tricoplanar, !isarea facets w/ ->keepcentrum */
    # }f;
    center::Ptr{QHcoordT}  # set according to qh.CENTERtype */
                           # qh_ASnone:    no center (not MERGING) */
                           # qh_AScentrum: centrum for testing convexity (qh_getcentrum) */
                           #               assumed qh_AScentrum while merging */
                           # qh_ASvoronoi: Voronoi center (qh_facetcenter) */
                           # after constructing the hull, it may be changed (qh_clearcenter) */
                           # if tricoplanar and !keepcentrum, shared with a neighbor */
    previous::Ptr{QHfacetT{HD}} # previous facet in the facet_list or NULL, for C++ interface */
    next::Ptr{QHfacetT{HD}}     # next facet in the facet_list or facet_tail */
    vertices::Ptr{QHsetT{Ptr{QHvertexT{HD}}}}   # vertices for this facet, inverse sorted by ID
                            # if simplicial, 1st vertex was apex/furthest
                            # qh_reduce_vertices removes extraneous vertices via qh_remove_extravertices
                            # if f.visible, vertices may be on qh.del_vertices */
    ridges::Ptr{QHsetT}     # explicit ridges for nonsimplicial facets or nonsimplicial neighbors.
                            # For simplicial facets, neighbors define the ridges
                            # qh_makeridges() converts simplicial facets by creating ridges prior to merging
                            # If qh.NEWtentative, new facets have horizon ridge, but not vice versa
                            # if f.visible && qh.NEWfacets, ridges is empty */
    neighbors::Ptr{QHsetT{Ptr{QHfacetT{HD}}}}  # neighbors of the facet.  Neighbors may be f.visible
                            # If simplicial, the kth neighbor is opposite the kth vertex and the
                            # first neighbor is the horizon facet for the first vertex.
                            # dupridges marked by qh_DUPLICATEridge (0x01) and qh_MERGEridge (0x02)
                            # if f.visible && qh.NEWfacets, neighbors is empty */
    outsideset::Ptr{QHsetT} # set of points outside this facet
                            # if non-empty, last point is furthest
                            # if NARROWhull, includes coplanars (less than qh.MINoutside) for partitioning*/
    coplanarset::Ptr{QHsetT} # set of points coplanar with this facet
                             # >= qh.min_vertex and <= facet->max_outside
                             # a point is assigned to the furthest facet
                             # if non-empty, last point is furthest away */
    visitid::QHuintT         # visit_id, for visiting all neighbors, all uses are independent */
    id::QHuintT              # unique identifier from qh.facet_id, 1..qh.facet_id, 0 is sentinel, printed as 'f%d' */
    flags::QHuintT

    # unsigned int nummerge:9; /* number of merges */
    # define qh_MAXnummerge 511 /* 2^9-1 */
    #                        /* 23 flags (at most 23 due to nummerge), printed by "flags:" in io_r.c */
    # flagT    tricoplanar:1; /* True if TRIangulate and simplicial and coplanar with a neighbor */
    #                      /*   all tricoplanars share the same apex */
    #                      /*   all tricoplanars share the same ->center, ->normal, ->offset, ->maxoutside */
    #                      /*     ->keepcentrum is true for the owner.  It has the ->coplanareset */
    #                      /*   if ->degenerate, does not span facet (one logical ridge) */
    #                      /*   during qh_triangulate, f.trivisible points to original facet */
    # flagT    newfacet:1;  /* True if facet on qh.newfacet_list (new/qh.first_newfacet or merged) */
    # flagT    visible:1;   /* True if visible facet (will be deleted) */
    # flagT    toporient:1; /* True if created with top orientation
    #                       after merging, use ridge orientation */
    # flagT    simplicial:1;/* True if simplicial facet, ->ridges may be implicit */
    # flagT    seen:1;      /* used to perform operations only once, like visitid */
    # flagT    seen2:1;     /* used to perform operations only once, like visitid */
    # flagT    flipped:1;   /* True if facet is flipped */
    # flagT    upperdelaunay:1; /* True if facet is upper envelope of Delaunay triangulation */
    # flagT    notfurthest:1; /* True if last point of outsideset is not furthest */
    #
    # /*-------- flags primarily for output ---------*/
    # flagT    good:1;      /* True if a facet marked good for output */
    # flagT    isarea:1;    /* True if facet->f.area is defined */

    # /*-------- flags for merging ------------------*/
    # flagT    dupridge:1;  /* True if facet has one or more dupridge in a new facet (qh_matchneighbor),
    #                         a dupridge has a subridge shared by more than one new facet */
    # flagT    mergeridge:1; /* True if facet or neighbor has a qh_MERGEridge (qh_mark_dupridges)
    #                        ->normal defined for mergeridge and mergeridge2 */
    # flagT    mergeridge2:1; /* True if neighbor has a qh_MERGEridge (qh_mark_dupridges) */
    # flagT    coplanarhorizon:1;  /* True if horizon facet is coplanar at last use */
    # flagT     mergehorizon:1; /* True if will merge into horizon (its first neighbor w/ f.coplanarhorizon). */
    # flagT     cycledone:1;/* True if mergecycle_all already done */
    # flagT    tested:1;    /* True if facet convexity has been tested (false after merge */
    # flagT    keepcentrum:1; /* True if keep old centrum after a merge, or marks owner for ->tricoplanar
    #                          Set by qh_updatetested if more than qh_MAXnewcentrum extra vertices
    #                          Set by qh_mergefacet if |maxdist| > qh.WIDEfacet */
    # flagT    newmerge:1;  /* True if facet is newly merged for reducevertices */
    # flagT    degenerate:1; /* True if facet is degenerate (degen_mergeset or ->tricoplanar) */
    # flagT    redundant:1;  /* True if facet is redundant (degen_mergeset)
    #                     Maybe merge degenerate and redundant to gain another flag */
end


# Iteration for facet list
function iterate(first_fct::QHfacetT{HD}) where HD
    return (first_fct, first_fct.next_ptr)    
end

function iterate(first_fct::QHfacetT{HD}, next_fct_ptr::Ptr{QHfacetT{HD}}) where HD
    next_fct = unsafe_load(next_fct_ptr)
    ## ID=0 is dummy facet that indicates end of the list
    if next_fct.id == QHuintT(0)
        return nothing
    else
        return (next_fct, next_fct.next_ptr)
    end
end

# Iteration for vertex list
function iterate(first_vtx::QHvertexT{HD}) where HD
    return (first_vtx, first_vtx.next_ptr)    
end

function iterate(first_vtx::QHvertexT{HD}, next_vtx_ptr::Ptr{QHvertexT{HD}}) where HD
    next_vtx = unsafe_load(next_vtx_ptr)
    ## ID=0 is dummy vertext that indicates end of the list
    if next_vtx.id == QHuintT(0)
        return nothing
    else
        return (next_vtx, next_vtx.next_ptr)
    end
end


# Iterate QHsetT with pointer types
function iterate(set::QHsetT{T}) where T<:Ptr{<:QHsetelemT}

    if length(set.e) == 0
        return nothing
    else
        return (unsafe_load(set.e[1]), 2)
    end
end

function iterate(set::QHsetT{T}, idx::Int) where T<:Ptr{<:QHsetelemT}
    if idx > length(set.e)
        return nothing
    else
        return (unsafe_load(set.e[idx]), idx+1)
    end
end


# Build convex hull from a set points
struct ConvexHull
    qh_ptr::Ptr{qhT}
    points::Matrix{QHcoordT}
    vertices::Vector{QHuintT}
    simplices::Matrix{QHuintT}
    area::QHrealT
    volume::QHrealT
    max_bound::Vector{QHrealT}
    min_bound::Vector{QHrealT}
    
    function ConvexHull(pnts::Matrix{QHcoordT}, qhull_options::Vector{AbstractString}=Vector{AbstractString}())
        qh_ptr = qh_alloc_qh()

        pushfirst!(qhull_options, "Qt")
        
        if size(pnts, 2)>=5
            push!(qhull_options, "Qx")
        end

        # make options string
        qh_opts_str = foldl((l,r)->l*" "*r, qhull_options)

        # calculate new qhull
        res = qh_new_qhull(qh_ptr, pnts, qh_opts_str)

        
        hd = qh_get_hull_dim(qh_ptr)

        if hd == 2
            vertices = qh_get_extremes_2d(qh_ptr)
        else
            vertices = unique((vtx)->qh_pointid(qh_ptr, vtx.point_ptr)+1)
        end
        
        # collect convex hull points
        # for some reason using hd from above in Val(hd) crashes Julia
        simplices = qh_get_convex_hull_pnts(qh_ptr, Val(size(pnts,1)))

        # calculate total area and volume
        qh_getarea(qh_ptr, Val(size(pnts,1)))
        area = qh_get_totarea(qh_ptr)
        vol = qh_get_totvol(qh_ptr)

        # max and min bounds
        max_bound = maximum(pnts, dims=2)[:]
        min_bound = minimum(pnts, dims=2)[:]

        # the new Qhull value
        new(qh_ptr, pnts, vertices, simplices, area, vol, max_bound, min_bound)
    end    
end

# calculate new convex hull from the given points and Qhull options
function qh_new_qhull(qh::Ptr{qhT}, pnts::StridedMatrix{Float64}, opts::String)
    ok = ccall((:qh_new_qhull, qh_lib), Cint,
               (Ptr{qhT}, Cint, Cint, Ref{QHcoordT}, QHboolT, Ptr{QHcharT}, Ptr{QHfileT}, Ptr{QHfileT}),
               qh, size(pnts, 1), size(pnts, 2), pnts, false, "qhull " * opts, C_NULL, C_NULL)
    
    return ok
end

function qh_triangulate(qh::Ptr{qhT})
    ccall((:qh_triangulate, qh_lib), Cvoid, (Ptr{qhT},), qh)
end

function qh_getarea(qh::Ptr{qhT}, ::Val{HD}) where HD
    ccall((:qh_getarea, qh_lib), Cvoid, (Ptr{qhT}, Ptr{QHfacetT}), qh, qh_get_facet_list_ptr(qh, Val(HD)))
end
    
# retrieve Qhull internal point id
function qh_pointid(qh::Ptr{qhT}, pnt::Ptr{NTuple{N, QHpointT}}) where N
    id = ccall((:qh_pointid, qh_lib), Cuint,
               (Ptr{qhT}, Ptr{QHpointT}), qh, pnt)
    return id
end

# retrieve Qhull set size
function qh_setsize(qh::Ptr{qhT}, set::Ptr{QHsetT{T}}) where T<:Union{QHintT, Ptr{<:QHsetelemT}}
    ccall((:qh_setsize, qh_lib), Cint, (Ptr{qhT}, Ptr{QHsetT}), qh, set)
end


# GETTER accessors for plain data types
for (T, getter) in ((:QHintT, :qh_get_hull_dim), (:QHintT, :qh_get_num_facets), (:QHintT, :qh_get_num_points),
                    (:QHintT, :qh_get_num_vertices), (:QHintT, :qh_get_visit_id), (:QHintT, :qh_get_vertex_visit),
                    (:QHrealT, :qh_get_totarea), (:QHrealT, :qh_get_totvol))
    @eval begin
        function ($getter)(qh::Ptr{qhT})
            ccall(($(QuoteNode(getter)), qh_lib), $T, (Ptr{qhT},), qh)
        end
    end
end

# GETTER for facet list pointer
function qh_get_facet_list_ptr(qh::Ptr{qhT}, ::Val{N}) where N
    ccall((:qh_get_facet_list, qh_lib), Ptr{QHfacetT{N}}, (Ptr{qhT},), qh)
end

# GETTER for facet list as Julia QHfacetT type
@noinline function qh_get_facet_list(qh::Ptr{qhT}, ::Val{N}) where N
    local ptr

    # this is workaround for Julia crashes that seem to happen if Val(N) has N defined in runtime
    if N==1
        ptr = qh_get_facet_list_ptr(qh, Val(1))
    elseif N==2        
        ptr = qh_get_facet_list_ptr(qh, Val(2))
    elseif N==3
        ptr = qh_get_facet_list_ptr(qh, Val(3))
    elseif N==4
        ptr = qh_get_facet_list_ptr(qh, Val(4))
    elseif N==5
        ptr = qh_get_facet_list_ptr(qh, Val(5))
    else
        error("Too large hull dimension")
    end

    if ptr != C_NULL
        return unsafe_load(ptr)
    else
        return nothing
    end        
end

# GETTER for facet list pointer
function qh_get_vertex_list_ptr(qh::Ptr{qhT}, ::Val{N}) where N
    ccall((:qh_get_vertex_list, qh_lib), Ptr{QHvertexT{N}}, (Ptr{qhT},), qh)
end

# GETTER for facet list as Julia QHfacetT type
@noinline function qh_get_vertex_list(qh::Ptr{qhT}, ::Val{N}) where N
    local ptr
    # this is workaround for Julia crashes that seem to happen if Val(N) has N defined in runtime
    if N==1
        ptr = qh_get_vertex_list_ptr(qh, Val(1))
    elseif N==2        
        ptr = qh_get_vertex_list_ptr(qh, Val(2))
    elseif N==3
        ptr = qh_get_vertex_list_ptr(qh, Val(3))
    elseif N==4
        ptr = qh_get_vertex_list_ptr(qh, Val(4))
    elseif N==5
        ptr = qh_get_vertex_list_ptr(qh, Val(5))
    else
        error("Too large hull dimension")
    end
    if ptr != C_NULL
        return unsafe_load(ptr)
    else
        return nothing
    end        
end


function Base.getproperty(qh::ConvexHull, fld::Symbol)
    if fld === :hull_dim
        return qh_get_hull_dim(qh.qh_ptr)
    elseif fld === :num_facets
        return qh_get_num_facets(qh.qh_ptr)
    elseif fld === :num_points
        return qh_get_num_points(qh.qh_ptr)
    elseif fld === :num_vertices
        return qh_get_num_vertices(qh.qh_ptr)
    elseif fld === :visit_id
        return qh_get_visit_id(qh.qh_ptr)
    elseif fld === :vertex_visit
        return qh_get_vertex_visit(qh.qh_ptr)
    elseif fld === :facet_list
        # Hull dimension given to facet type so that hull dimension array size is known
        return qh_get_facet_list(qh.qh_ptr, Val(qh.hull_dim))
    elseif fld === :vertex_list
        # Hull dimension given to facet type so that hull dimension array size is known
        return qh_get_vertex_list(qh.qh_ptr, Val(qh.hull_dim))
    else
        return getfield(qh, fld)
    end
end

function Base.getproperty(fct::QHfacetT{HD}, fld::Symbol) where HD
    if fld === :next
        ptr = getfield(fct, :next)
        if ptr == C_NULL
            return nothing
        else
            return unsafe_load(ptr)
        end
    elseif fld === :next_ptr
        return getfield(fct, :next)
    elseif fld === :vertices
        return QHsetT{Ptr{QHvertexT{HD}}}(fct.vertices_ptr)
    elseif fld === :vertices_ptr
        return getfield(fct, :vertices)
    elseif fld === :neighbors
        return QHsetT{Ptr{QHfacetT{HD}}}(fct.neighbors_ptr)
    elseif fld === :neighbors_ptr
        return getfield(fct, :neighbors)
    elseif fld == :toporient
        return QHboolT( (getfield(fct, :flags)>>12)&1)  # toporient is 13th bit in the flags field
    else
        return getfield(fct, fld)
    end
end


function Base.setproperty!(fct::QHfacetT{HD}, fld::Symbol, value) where HD
    setfield!(fct, fld, value)
    # store back to qhull, pointer to self is obtained (somewhat dangerously) utilizing the facet linked list
    if (fct.id != 0) # ID=0 is dummy facet, with next=C_NULL (so the facet cannot be updated in this case)
        unsafe_store!(fct.next.previous, fct)
    end
    return value
end



function Base.getproperty(vtx::QHvertexT, fld::Symbol)
    if fld === :next
        ptr = getfield(vtx, :next)
        if ptr == C_NULL
            return nothing
        else
            return unsafe_load(ptr)
        end
    elseif fld === :next_ptr
        return getfield(vtx, :next)
    elseif fld === :point
        ptr = getfield(vtx, :point)
        if ptr == C_NULL
            return nothing
        else
            return unsafe_load(ptr)
        end
    elseif fld === :point_ptr
        return getfield(vtx, :point)
    else
        return getfield(vtx, fld)
    end
end

function Base.setproperty!(vtx::QHvertexT{HD}, fld::Symbol, value) where HD
    setfield!(vtx, fld, value)
    # store back to qhull, pointer to self is obtained (somewhat dangerously) utilizing the vertex linked list
    if (vtx.id != 0) # ID=0 is dummy vertex, with next=C_NULL (so the vertex cannot be updated in this case)
        unsafe_store!(vtx.next.previous, vtx)
    end
end

# get calculated convex hull points as Julia Int Array
function qh_get_convex_hull_pnts(qh_ptr::Ptr{qhT}, ::Val{HD}) where HD
    n_facets = qh_get_num_facets(qh_ptr)
    
    facet_list = qh_get_facet_list(qh_ptr, Val(HD))

    # convex hull point ids (i.e. indexes to input points to qhull)
    pnts = Matrix{QHuintT}(undef, HD, n_facets)
    
    facet_ix = 1
    for facet in facet_list
        vtxSet = facet.vertices

        vtx_ix = 1
        for vtx in vtxSet            
            pnt_id = qh_pointid(qh_ptr, vtx.point_ptr)

            # +1 to change to 1-based index for Julia from C 0-based index
            pnts[vtx_ix, facet_ix] = pnt_id + 1 
            vtx_ix+=1
        end
        facet_ix+=1
    end
    
    return pnts
end

# get calculated convex hull vertices as Julia Int Array
function qh_get_convex_hull_vertices(qh_ptr::Ptr{qhT}, ::Val{HD}) where HD
    n_vertices = qh_get_num_vertices(qh_ptr)

    vertex_list = qh_get_vertex_list(qh_ptr, Val(HD))

    # convex hull point ids (i.e. indexes to input points to qhull)
    vertices = Vector{QHuintT}(undef, n_vertices)
    
    vertex_ix = 1
    for vtx in vertex_list
        pnt_id = qh_pointid(qh_ptr, vtx.point_ptr)
        vertices[vertex_ix] = pnt_id + 1
        vertex_ix+=1
    end
    
    return vertices
end


# Below function "qh_get_extremes_2d" adapted from from Qhull/io.c and
# Scipy/_qhull.pyx/get_extremes_2d with the below license from _qhull.pyx/Scipy
#
# Copyright (C)  Pauli Virtanen, 2010.
#
# Distributed under the same BSD license as Scipy.
#
# 
# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided
#   with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE


function qh_get_extremes_2d(qh_ptr::Ptr{qhT}) 

    # qhull io.c has the below call, Scipy does not
    # This is called from ConvexHull constructor, hence assumed visit_id is valid
    # qh_countfacets(facetlist, facets, printall, &numfacets, &numsimplicial,
    #    &totneighbors, &numridges, &numcoplanars, &numtricoplanars); /* marks qh visit_id */
    # vertices= qh_facetvertices(facetlist, facets, printall);
    # qh_fprintf(fp, 9088, "%d\n", qh_setsize(vertices));
    # qh_settempfree(&vertices);
    
    if (qh_get_num_facets(qh_ptr) == 0)
        return Vector{QHintT}()
    end
    
    # qhull/Scipy update the internal qhT structure fields for the visit ids
    # as the values are only read in the rest of the code, local variables used here
    # (assumed the qh internal value update is not necessary)
    qh_vertex_visit = qh_get_vertex_visit(qh_ptr) + 1
    qh_visit_id = qh_get_visit_id(qh_ptr) + 1

    # Init result array
    extremes = zeros(QHintT, 100)
    n_extremes = 0

    # get first facet in facet list
    facet = qh_get_facet_list(qh_ptr, Val(qh_get_hull_dim(qh_ptr)))

    # use facet id instead of pointer comparision for ending the while loop
    start_facet_id = facet.id

    while !isnothing(facet)
        if facet.visitid == qh_visit_id
            throw(ErrorException("Internal Qhull error, loop in facet list"))
        end
        
        if xor(facet.toporient, qh_ORIENTclock) != 0
            vertexA = facet.vertices[1] 
            vertexB = facet.vertices[2]  
            nextfacet = facet.neighbors[1]
        else
            vertexA = facet.vertices[2]  
            vertexB = facet.vertices[1]  
            nextfacet = facet.neighbors[2]
        end
        
        # check if result array needs resizing
        if n_extremes + 2 > length(extremes)
            resize!(extremes, 2*length(extremes)+1)
        end
        if facet.visitid != 0  # qhull has this check, Scipy does not
            if (vertexA.visitid != qh_vertex_visit)
                # this updates also internal qhull vertex state
                vertexA.visitid = QHuintT(qh_vertex_visit)
                n_extremes += 1            
                extremes[n_extremes] = qh_pointid(qh_ptr, vertexA.point_ptr) + 1 # +1 to change to 1-based index
            end
            if vertexB.visitid != qh_vertex_visit
                vertexB.visitid = QHuintT(qh_vertex_visit)
                n_extremes += 1            
                extremes[n_extremes] = qh_pointid(qh_ptr, vertexB.point_ptr) + 1 # +1 to change to 1-based index
            end
        end
        # this updates also internal qhull facet state
        facet.visitid = QHuintT(qh_visit_id);
        facet = nextfacet

        if facet.id == start_facet_id
            break
        end
    end
    
    resize!(extremes, n_extremes)
    return extremes
    
end

end
