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

module DirectQhull

import Base.getproperty
import Base.setproperty!
import Base.getindex

import Qhull_jll
import DirectQhullHelper_jll

const libqhull_r = Qhull_jll.get_libqhull_r_path()
const libqhull_helper = DirectQhullHelper_jll.libDirectQhullHelper

# define Qhull types
QHboolT = Cuint
QHrealT = Cdouble
QHint = Cint
QHprintT = Cint
QHchar = Cchar
QHjmpbufT = Ptr{Cuchar}
QHfileT = Cvoid
QHuint = Cuint
QHridgeT = Ptr{Cuchar}
QHulong = Culong
QHcenterT = Cint
QHdouble = Cdouble
QHvoid = Cvoid
QHmemT = Ptr{Cuchar}
QHstatT = Ptr{Cuchar}

# mutable structs for accessing Qhull internals
mutable struct QHflagT
    flag::Cuint
end

mutable struct QHqhT
    qh::Vector{UInt8}
end

mutable struct QHfacetT
    ptr::Ptr{Vector{UInt8}}
end

mutable struct QHvertexT
    ptr::Ptr{Vector{UInt8}}
end

mutable struct QHpointT
    ptr::Ptr{Vector{QHrealT}}
end

mutable struct QHcoordT
    ptr::Ptr{Vector{QHrealT}}
end

mutable struct QHsetT{T<:Union{QHvertexT, QHfacetT}}
    ptr::Ptr{Vector{UInt8}}
end

# retrieve "struct qhT" c-struct size in bytes (used to reserve memory when accessing Qhull directly)
function qh_qhT_size()
    ccall((:jl_qhull_qhT_size, libqhull_helper), Csize_t,
          (),)
end

# retrieve "struct facetT" c-struct size in bytes (used to reserve memory when accessing Qhull directly)
function qh_facetT_size()
    ccall((:jl_qhull_facetT_size, libqhull_helper), Csize_t,
          (),)
end

# zero "struct qhT", 
function qh_zero(qh::QHqhT)
    ccall((:qh_zero, libqhull_r), Cvoid,
          (Ptr{Vector{UInt8}}, Ptr{Cvoid}),
          pointer(qh.qh), C_NULL)
end

# reserve memory for accessing Qhull directly
function qh_init()
    QHqhT(Vector{UInt8}(undef, qh_qhT_size()))
end

# calculate new convex hull from the given points and Qhull options
function qh_new_qhull(qh::QHqhT, pnts::StridedMatrix{Float64}, opts::String)
    ok = ccall((:qh_new_qhull, libqhull_r), Cint,
          (Ptr{Vector{UInt8}}, Cint, Cint, Ptr{QHcoordT}, QHboolT, Ptr{QHchar}, Ptr{QHfileT}, Ptr{QHfileT}),
               pointer(qh.qh), size(pnts,1), size(pnts,2), pointer(pnts), false, "qhull  " * opts, C_NULL, C_NULL)
    
    return ok
end

# retrieve Qhull internal point id
function qh_point_id(qh::QHqhT, pnt::QHpointT)
    ok = ccall((:qh_pointid, libqhull_r), Cint,
          (Ptr{Vector{UInt8}}, Ptr{Vector{UInt8}}),
          pointer(qh.qh), pnt.ptr)
    return ok
end

# call Qhull findgood_all 
function qh_findgood_all(qh::QHqhT)
    ccall((:qh_findgood_all, libqhull_r), Cvoid,
          (Ptr{Vector{UInt8}}, Ptr{Vector{UInt8}}),
          pointer(qh.qh), qh.facet_list.ptr)
end

# call Qhull setsize
function qh_setsize(qh::QHqhT, set::QHsetT)
    ccall((:qh_setsize, libqhull_r), Cint,
          (Ptr{Vector{UInt8}}, Ptr{Vector{UInt8}}),
          pointer(qh.qh), set.ptr)
end

# call Qhull setvoronoi_all
function qh_setvoronoi_all(qh::QHqhT)
    ccall((:qh_setvoronoi_all, libqhull_r), Cvoid,
          (Ptr{Vector{UInt8}},),
          pointer(qh.qh))
end

# call Qhull order vertexneighbours
function qh_order_vertexneighbors(qh::QHqhT, vtx::QHvertexT)
    ccall((:qh_order_vertexneighbors, libqhull_r), Cvoid,
          (Ptr{Vector{UInt8}},Ptr{Vector{UInt8}}),
          pointer(qh.qh), vtx.ptr)
end

# get calculated convex hull points as Julia Int Array
function qh_get_convex_hull_pnts(qh::QHqhT)
    n_vtx = qh.num_vertices
    n_facets = qh.num_facets
    facet = qh.facet_list

    hull_dim = qh.hull_dim 

    pnts = Array{Int,2}(undef, n_facets, hull_dim)
    vtxSet = facet.vertices
    pnt_ix = 1
    for facetI=1:n_facets
        vtxSet = facet.vertices
        vtxI = 1
        while vtxI <= vtxSet.maxsize && vtxSet[vtxI].ptr != C_NULL                                
            pnt_id = qh_point_id(qh, vtxSet[vtxI].point)
            pnts[pnt_ix, vtxI] = pnt_id
            vtxI += 1
        end
        pnt_ix += 1
        facet = facet.next
    end

    return pnts
end

# get calculated dalaunay points as Julia Int Array
function qh_get_delaunay_pnts(qh::QHqhT)
    n_vtx = qh.num_vertices
    n_facets = 0
    facet = qh.facet_list

    for facetI=1:qh.num_facets
        if facet.upperdelaunay  == 0
            n_facets += 1
        end
        facet = facet.next
    end
    
    hull_dim = qh.hull_dim 

    pnts = Array{Int,2}(undef, n_facets, hull_dim)
    vtxSet = facet.vertices
    facet = qh.facet_list
    pnt_ix = 1
    for facetI=1:qh.num_facets
        if facet.upperdelaunay  == 0
            vtxSet = facet.vertices
            vtxI = 1
            while vtxI <= vtxSet.maxsize && vtxSet[vtxI].ptr != C_NULL                                
                pnt_id = qh_point_id(qh, vtxSet[vtxI].point)
                pnts[pnt_ix, vtxI] = pnt_id
                vtxI += 1
            end
            pnt_ix += 1
        end
        facet = facet.next
    end

    return pnts
end

# get calculated voroin points as Julia array
function qh_get_voronoi_pnts(qh::QHqhT)

    qh_findgood_all(qh)

    num_voronoi_regions = qh.num_vertices - qh_setsize(qh, qh.del_vertices)

    num_voronoi_vertices = qh.num_good
    
    qh_setvoronoi_all(qh)

    facet = qh.facet_list
    while facet.ptr != qh.facet_tail.ptr
       # facet.seen = false
        facet = facet.next
    end

    ni = zeros(Int, num_voronoi_regions)

    k = 1
    vertex = qh.vertex_list
    while vertex.ptr != qh.vertex_tail.ptr
        if qh.hull_dim == 3
            qh_order_vertexneighbors(qh, vertex)
        end
        
        infinity_seen = false

        neighborSet = vertex.neighbors
        for neighborI = 1:qh_setsize(qh, neighborSet)
            neighbor = neighborSet[neighborI]
            if neighbor.upperdelaunay  != 0
                if infinity_seen == false
                    infinity_seen = true
                    ni[k] += 1
                end
            else
             #   neighbor.seen = true
                ni[k] += 1
            end
        end
        k += 1
        vertex = vertex.next
    end

    nr = (qh.num_points > num_voronoi_regions) ? qh.num_points : num_voronoi_regions

    at_inf = zeros(Bool, nr, 1)
    F = zeros(num_voronoi_vertices+1, qh.input_dim)
    F[1, :] .= Inf

    C = Array{Any,2}(undef, nr, 1)
    fill!(C, Array{Float64,2}(undef,0,0))
    facet = qh.facet_list
    for facetI=1:qh.num_facets
        facet.seen = false
        facet = facet.next
    end

    i = 0
    k = 1

    vertex = qh.vertex_list
    while vertex.ptr != qh.vertex_tail.ptr
        if qh.hull_dim == 3
            qh_order_vertexneighbors(qh, vertex)
        end
        infinity_seen = false
        idx = qh_point_id(qh, vertex.point)
        num_vertices = ni[k]
        k += 1

        if num_vertices == 1
            continue
        end
        facet_list = zeros(Int, num_vertices)

        m = 1

        neighborSet = vertex.neighbors
        for neighborI = 1:qh_setsize(qh, neighborSet)
            neighbor = neighborSet[neighborI]
            if neighbor.upperdelaunay  != 0
                if infinity_seen == false
                    infinity_seen = true
                    facet_list[m] = 1
                    m += 1
                    at_inf[idx+1] = true
                end
            else
                if neighbor.seen == false
                    i += 1
                    for d = 1:qh.input_dim
                        F[i+1, d] = neighbor.center[d]
                    end
                    neighbor.seen = true
                    neighbor.visitid = i
                end
                facet_list[m] = neighbor.visitid + 1
                m += 1
            end
        end
        C[idx+1] = facet_list
        vertex = vertex.next
    end

    return (F, C, at_inf)
                
end

# getproperty/setproperty methods to get/set qhT struct internal fields
function getproperty(qh::QHqhT, fld::Symbol)
    if fld == :qh
        return invoke(getproperty, Tuple{Any, Symbol}, qh, :qh)
    else
        (offset, qhT) = _qhT_defs[fld]
        return unsafe_load(Ptr{qhT}(pointer(qh.qh) + _qhT_offsets[offset+1]))
    end
end


function setproperty!(qh::QHqhT, fld::Symbol, val)
    if fld == :qh
        return invoke(setproperty, Tuple{Any, Symbol, Any}, qh, :qh, val)
    else
        (offset, qhT) = _qhT_defs[fld]
        return unsafe_store!(Ptr{qhT}(pointer(qh.qh) + _qhT_offsets[offset+1]), qhT(val))
    end
end

function getproperty(facet::QHfacetT, fld::Symbol)
    if fld == :ptr
        return invoke(getproperty, Tuple{Any, Symbol}, facet, :ptr)
    else
        (offset, facetT) = _facetT_defs[fld]
        if facetT == QHflagT
            flag_offset = (offset-floor(offset)).num
            offset = Int(floor(offset))
            val = (unsafe_load(Ptr{facetT}(facet.ptr + _facetT_offsets[offset+1])).flag >> flag_offset) & 0x1
            return val
        else
            offset = Int(floor(offset))
            return unsafe_load(Ptr{facetT}(facet.ptr + _facetT_offsets[offset+1]))
        end
    end
end

function setproperty!(facet::QHfacetT, fld::Symbol, val)
    local load_offset
    if fld == :ptr
        return invoke(setproperty, Tuple{Any, Symbol, Any}, facet, :ptr, val)
    else
        (offset, facetT) = _facetT_defs[fld]
        if facetT == QHflagT
            load_offset = Int(floor(offset))
            curr_val = unsafe_load(Ptr{facetT}(facet.ptr + _facetT_offsets[load_offset+1])).flag
            flag_offset = (offset-floor(offset)).num
            val = (curr_val & (xor(1 << flag_offset,0xffffffff))) | (val << flag_offset)
        else
            load_offset = Int(floor(offset))
        end
        
        return unsafe_store!(Ptr{facetT}(facet.ptr + _facetT_offsets[load_offset+1]), facetT(val))

    end
end

function getproperty(vertex::QHvertexT, fld::Symbol)
    if fld == :ptr
        return invoke(getproperty, Tuple{Any, Symbol}, vertex, :ptr)
    else
        (offset, fldT) = _vertexT_defs[fld]
        if fld == :point
            return unsafe_load(Ptr{fldT}(vertex.ptr + _vertexT_offsets[offset+1]))
        else
            return unsafe_load(Ptr{fldT}(vertex.ptr + _vertexT_offsets[offset+1]))
        end
    end
end

function getproperty(vtxSet::QHsetT{T}, fld::Symbol) where T <: Union{QHvertexT, QHfacetT}
    if fld == :ptr
        return invoke(getproperty, Tuple{Any, Symbol}, vtxSet, :ptr)
    elseif fld == :maxsize
        return unsafe_load(Ptr{Cint}(vtxSet.ptr))
    else
        error("Invalid field.")
    end
end

function getindex(vtxSet::QHsetT{T}, I::Int) where T <: Union{QHvertexT, QHfacetT}
    if I > vtxSet.maxsize
        error("out of bounds.")
    end

    offset = I*_setT_offsets[2]
    return unsafe_load(Ptr{T}(vtxSet.ptr+offset))
    
end

function getindex(pnt::QHpointT, I::Int)
    offset = (I-1)*sizeof(Float64)
    return unsafe_load(Ptr{Float64}(pnt.ptr+offset))
end

function getindex(pnt::QHcoordT, I::Int)
    offset = (I-1)*sizeof(Float64)
    return unsafe_load(Ptr{Float64}(pnt.ptr+offset))
end


# methods/settings to access Qhull internals directly

const _qhT_offsets = Vector{Csize_t}(undef, 257)
const _facetT_offsets = Vector{Csize_t}(undef, 44)
const _vertexT_offsets = Vector{Csize_t}(undef, 12)
const _setT_offsets = Vector{Csize_t}(undef, 2)

function _set_qhT_offsets()
    ccall((:jl_qhull_qhT_offsets, libqhull_helper), Cvoid,
          (Ptr{Vector{Csize_t}},), pointer(_qhT_offsets))
end
function _set_facetT_offsets()
    ccall((:jl_qhull_facetT_offsets, libqhull_helper), Cvoid,
          (Ptr{Vector{Csize_t}},), pointer(_facetT_offsets))
end
function _set_vertexT_offsets()
    ccall((:jl_qhull_vertexT_offsets, libqhull_helper), Cvoid,
          (Ptr{Vector{Csize_t}},), pointer(_vertexT_offsets))
end

function _set_setT_offsets()
    ccall((:jl_qhull_setT_offsets, libqhull_helper), Cvoid,
          (Ptr{Vector{Csize_t}},), pointer(_setT_offsets))
end

_set_qhT_offsets()
_set_facetT_offsets()
_set_vertexT_offsets()
_set_setT_offsets()

_qhT_defs = Dict{Symbol, Tuple{Int,DataType}}([(:ALLpoints => (0, QHboolT)),
                                             (:ALLOWshort => (1, QHboolT)),
(:ALLOWwarning => (2, QHboolT)),
(:ALLOWwide => (3, QHboolT)),
(:ANGLEmerge => (4, QHboolT)),
(:APPROXhull => (5, QHboolT)),
(:MINoutside => (6, QHrealT)),
(:ANNOTATEoutput => (7, QHboolT)),
(:ATinfinity => (8, QHboolT)),
(:AVOIDold => (9, QHboolT)),
(:BESToutside => (10, QHboolT)),
(:CDDinput => (11, QHboolT)),
(:CDDoutput => (12, QHboolT)),
(:CHECKduplicates => (13, QHboolT)),
(:CHECKfrequently => (14, QHboolT)),
(:premerge_cos => (15, QHrealT)),
(:postmerge_cos => (16, QHrealT)),
(:DELAUNAY => (17, QHboolT)),
(:DOintersections => (18, QHboolT)),
(:DROPdim => (19, QHint)),
(:FLUSHprint => (20, QHboolT)),
(:FORCEoutput => (21, QHboolT)),
(:GOODpoint => (22, QHint)),
(:GOODpointp => (23, Ptr{QHpointT})),
(:GOODthreshold => (24, QHboolT)),
(:GOODvertex => (25, QHint)),
(:GOODvertexp => (26, Ptr{QHpointT})),
(:HALFspace => (27, QHboolT)),
(:ISqhullQh => (28, QHboolT)),
(:IStracing => (29, QHint)),
(:KEEParea => (30, QHint)),
(:KEEPcoplanar => (31, QHboolT)),
(:KEEPinside => (32, QHboolT)),
(:KEEPmerge => (33, QHint)),
(:KEEPminArea => (34, QHrealT)),
(:MAXcoplanar => (35, QHrealT)),
(:MAXwide => (36, QHint)),
(:MERGEexact => (37, QHboolT)),
(:MERGEindependent => (38, QHboolT)),
(:MERGING => (39, QHboolT)),
(:premerge_centrum => (40, QHrealT)),
(:postmerge_centrum => (41, QHrealT)),
(:MERGEpinched => (42, QHboolT)),
(:MERGEvertices => (43, QHboolT)),
(:MINvisible => (44, QHrealT)),
(:NOnarrow => (45, QHboolT)),
(:NOnearinside => (46, QHboolT)),
(:NOpremerge => (47, QHboolT)),
(:ONLYgood => (48, QHboolT)),
(:ONLYmax => (49, QHboolT)),
(:PICKfurthest => (50, QHboolT)),
(:POSTmerge => (51, QHboolT)),
(:PREmerge => (52, QHboolT)),
(:PRINTcentrums => (53, QHboolT)),
(:PRINTcoplanar => (54, QHboolT)),
(:PRINTdim => (55, QHint)),
(:PRINTdots => (56, QHboolT)),
(:PRINTgood => (57, QHboolT)),
(:PRINTinner => (58, QHboolT)),
(:PRINTneighbors => (59, QHboolT)),
(:PRINTnoplanes => (60, QHboolT)),
(:PRINToptions1st => (61, QHboolT)),
(:PRINTouter => (62, QHboolT)),
(:PRINTprecision => (63, QHboolT)),
(:PRINTout => (64, QHprintT)),
(:PRINTridges => (65, QHboolT)),
(:PRINTspheres => (66, QHboolT)),
(:PRINTstatistics => (67, QHboolT)),
(:PRINTsummary => (68, QHboolT)),
(:PRINTtransparent => (69, QHboolT)),
(:PROJECTdelaunay => (70, QHboolT)),
(:PROJECTinput => (71, QHint)),
(:RANDOMdist => (72, QHboolT)),
(:RANDOMfactor => (73, QHrealT)),
(:RANDOMa => (74, QHrealT)),
(:RANDOMb => (75, QHrealT)),
(:RANDOMoutside => (76, QHboolT)),
(:REPORTfreq => (77, QHint)),
(:REPORTfreq2 => (78, QHint)),
(:RERUN => (79, QHint)),
(:ROTATErandom => (80, QHint)),
(:SCALEinput => (81, QHboolT)),
(:SCALElast => (82, QHboolT)),
(:SETroundoff => (83, QHboolT)),
(:SKIPcheckmax => (84, QHboolT)),
(:SKIPconvex => (85, QHboolT)),
(:SPLITthresholds => (86, QHboolT)),
(:STOPadd => (87, QHint)),
(:STOPcone => (88, QHint)),
(:STOPpoint => (89, QHint)),
(:TESTpoints => (90, QHint)),
(:TESTvneighbors => (91, QHboolT)),
(:TRACElevel => (92, QHint)),
(:TRACElastrun => (93, QHint)),
(:TRACEpoint => (94, QHint)),
(:TRACEdist => (95, QHrealT)),
(:TRACEmerge => (96, QHint)),
(:TRIangulate => (97, QHboolT)),
(:TRInormals => (98, QHboolT)),
(:UPPERdelaunay => (99, QHboolT)),
(:USEstdout => (100, QHboolT)),
(:VERIFYoutput => (101, QHboolT)),
(:VIRTUALmemory => (102, QHboolT)),
(:VORONOI => (103, QHboolT)),
(:AREAfactor => (104, QHrealT)),
(:DOcheckmax => (105, QHboolT)),
(:feasible_string => (106, Ptr{QHchar})),
(:feasible_point => (107, Ptr{QHcoordT})),
(:GETarea => (108, QHboolT)),
(:KEEPnearinside => (109, QHboolT)),
(:hull_dim => (110, QHint)),
(:input_dim => (111, QHint)),
(:num_points => (112, QHint)),
(:first_point => (113, Ptr{QHpointT})),
(:POINTSmalloc => (114, QHboolT)),
(:input_points => (115, Ptr{QHpointT})),
(:input_malloc => (116, QHboolT)),
(:qhull_command => (117, QHchar)),
(:qhull_commandsiz2 => (118, QHint)),
(:rbox_command => (119, QHchar)),
(:qhull_options => (120, QHchar)),
(:qhull_optionlen => (121, QHint)),
(:qhull_optionsiz => (122, QHint)),
(:qhull_optionsiz2 => (123, QHint)),
(:run_id => (124, QHint)),
(:VERTEXneighbors => (125, QHboolT)),
(:ZEROcentrum => (126, QHboolT)),
(:upper_threshold => (127, Ptr{QHrealT})),
(:lower_threshold => (128, Ptr{QHrealT})),
(:upper_bound => (129, Ptr{QHrealT})),
(:lower_bound => (130, Ptr{QHrealT})),
(:ANGLEround => (131, QHrealT)),
(:centrum_radius => (132, QHrealT)),
(:cos_max => (133, QHrealT)),
(:DISTround => (134, QHrealT)),
(:MAXabs_coord => (135, QHrealT)),
(:MAXlastcoord => (136, QHrealT)),
(:MAXoutside => (137, QHrealT)),
(:MAXsumcoord => (138, QHrealT)),
(:MAXwidth => (139, QHrealT)),
(:MINdenom_1 => (140, QHrealT)),
(:MINdenom => (141, QHrealT)),
(:MINdenom_1_2 => (142, QHrealT)),
(:MINdenom_2 => (143, QHrealT)),
(:MINlastcoord => (144, QHrealT)),
(:NEARzero => (145, Ptr{QHrealT})),
(:NEARinside => (146, QHrealT)),
(:ONEmerge => (147, QHrealT)),
(:outside_err => (148, QHrealT)),
(:WIDEfacet => (149, QHrealT)),
(:NARROWhull => (150, QHboolT)),
(:qhull => (151, QHchar)),
(:errexit => (152, QHjmpbufT)),
(:jmpXtra => (153, QHchar)),
(:restartexit => (154, QHjmpbufT)),
(:jmpXtra2 => (155, QHchar)),
(:fin => (156, Ptr{QHfileT})),
(:fout => (157, Ptr{QHfileT})),
(:ferr => (158, Ptr{QHfileT})),
(:interior_point => (159, Ptr{QHpointT})),
(:normal_size => (160, QHint)),
(:center_size => (161, QHint)),
(:TEMPsize => (162, QHint)),
(:facet_list => (163, QHfacetT)),
(:facet_tail => (164, QHfacetT)),
(:facet_next => (165, QHfacetT)),
(:newfacet_list => (166, QHfacetT)),
(:visible_list => (167, QHfacetT)),
(:num_visible => (168, QHint)),
(:tracefacet_id => (169, QHuint)),
(:tracefacet => (170, Ptr{QHfacetT})),
(:traceridge_id => (171, QHuint)),
(:traceridge => (172, Ptr{QHridgeT})),
(:tracevertex_id => (173, QHuint)),
(:tracevertex => (174, Ptr{QHvertexT})),
(:vertex_list => (175, QHvertexT)),
(:vertex_tail => (176, QHvertexT)),
(:newvertex_list => (177, Ptr{QHvertexT})),
(:num_facets => (178, QHint)),
(:num_vertices => (179, QHint)),
(:num_outside => (180, QHint)),
(:num_good => (181, QHint)),
(:facet_id => (182, QHuint)),
(:ridge_id => (183, QHuint)),
(:vertex_id => (184, QHuint)),
(:first_newfacet => (185, QHuint)),
(:hulltime => (186, QHulong)),
(:ALLOWrestart => (187, QHboolT)),
(:build_cnt => (188, QHint)),
(:CENTERtype => (189, QHcenterT)),
(:furthest_id => (190, QHint)),
(:last_errcode => (191, QHint)),
(:GOODclosest => (192, Ptr{QHfacetT})),
(:coplanar_apex => (193, Ptr{QHpointT})),
(:hasAreaVolume => (194, QHboolT)),
(:hasTriangulation => (195, QHboolT)),
(:isRenameVertex => (196, QHboolT)),
(:JOGGLEmax => (197, QHrealT)),
(:maxoutdone => (198, QHboolT)),
(:max_outside => (199, QHrealT)),
(:max_vertex => (200, QHrealT)),
(:min_vertex => (201, QHrealT)),
(:NEWfacets => (202, QHboolT)),
(:NEWtentative => (203, QHboolT)),
(:findbestnew => (204, QHboolT)),
(:findbest_notsharp => (205, QHboolT)),
(:NOerrexit => (206, QHboolT)),
(:PRINTcradius => (207, QHrealT)),
(:PRINTradius => (208, QHrealT)),
(:POSTmerging => (209, QHboolT)),
(:printoutvar => (210, QHint)),
(:printoutnum => (211, QHint)),
(:repart_facetid => (212, QHuint)),
(:retry_addpoint => (213, QHint)),
(:QHULLfinished => (214, QHboolT)),
(:totarea => (215, QHrealT)),
(:totvol => (216, QHrealT)),
(:visit_id => (217, QHuint)),
(:vertex_visit => (218, QHuint)),
(:WAScoplanar => (219, QHboolT)),
(:ZEROall_ok => (220, QHboolT)),
(:facet_mergeset => (221, Ptr{QHsetT})),
(:degen_mergeset => (222, Ptr{QHsetT})),
(:vertex_mergeset => (223, Ptr{QHsetT})),
(:hash_table => (224, Ptr{QHsetT})),
(:other_points => (225, Ptr{QHsetT})),
(:del_vertices => (226, QHsetT{QHvertexT})),
(:gm_matrix => (227, Ptr{QHcoordT})),
(:gm_row => (228, Ptr{Ptr{QHcoordT}})),
(:line => (229, Ptr{QHchar})),
(:maxline => (230, QHint)),
(:half_space => (231, Ptr{QHcoordT})),
(:temp_malloc => (232, Ptr{QHcoordT})),
(:ERREXITcalled => (233, QHboolT)),
(:firstcentrum => (234, QHboolT)),
(:old_randomdist => (235, QHboolT)),
(:coplanarfacetset => (236, Ptr{QHsetT})),
(:last_low => (237, QHrealT)),
(:last_high => (238, QHrealT)),
(:last_newhigh => (239, QHrealT)),
(:lastcpu => (240, QHrealT)),
(:lastfacets => (241, QHint)),
(:lastmerges => (242, QHint)),
(:lastplanes => (243, QHint)),
(:lastdist => (244, QHint)),
(:lastreport => (245, QHuint)),
(:mergereport => (246, QHint)),
(:old_tempstack => (247, Ptr{QHsetT})),
(:ridgeoutnum => (248, QHint)),
(:last_random => (249, QHint)),
(:rbox_errexit => (250, QHjmpbufT)),
(:jmpXtra3 => (251, QHchar)),
(:rbox_isinteger => (252, QHint)),
(:rbox_out_offset => (253, QHdouble)),
(:cpp_object => (254, QHvoid)),
(:qhmem => (255, QHmemT)),
(:qhstat => (256, QHstatT))])

_facetT_defs = Dict{Symbol, Tuple{Number,DataType}}([
(:furthestdist => (0, QHcoordT)),
(:maxoutside => (1, QHcoordT)),
(:offset => (2, QHcoordT)),
(:normal => (3, Ptr{QHcoordT})),
(:area => (4, QHrealT)),
(:replace => (5, QHfacetT)),
(:samecycle => (6, QHfacetT)),
(:newcycle => (7, QHfacetT)),
(:trivisible => (8,QHfacetT)),
(:triowner => (9, QHfacetT)),
(:center => (10, QHcoordT)),
(:previous => (11, QHfacetT)),
(:next => (12, QHfacetT)),
(:vertices => (13, QHsetT{QHvertexT})),
(:ridges => (14, Ptr{QHsetT})),
(:neighbors => (15, Ptr{QHsetT})),
(:outsideset => (16, Ptr{QHsetT})),
(:coplanarset => (17, Ptr{QHsetT})),
(:visitid => (18, QHuint)),
(:id => (19, QHuint)),
(:nummerge => (20, QHuint)),
(:tricoplanar => (21, QHflagT)),
(:newfacet => (22, QHflagT)),
(:visible => (23, QHflagT)),
(:toporient => (24, QHflagT)),
(:implicial => (25, QHflagT)),
(:seen => (26+14//33, QHflagT)),
(:seen2 => (27, QHflagT)),
(:flipped => (28, QHflagT)),
(:upperdelaunay => (29+17//33, QHflagT)),
(:notfurthest => (30, QHflagT)),
(:good => (31, QHflagT)),
(:isarea => (32, QHflagT)),
(:dupridge => (33, QHflagT)),
(:mergeridge => (34, QHflagT)),
(:mergeridge2 => (35, QHflagT)),
(:coplanarhorizon => (36, QHflagT)),
(:mergehorizon => (37, QHflagT)),
(:cycledone => (38, QHflagT)),
(:tested => (39, QHflagT)),
(:keepcentrum => (40, QHflagT)),
(:newmerge => (41, QHflagT)),
(:degenerate => (42, QHflagT)),
    (:redundant => (43, QHflagT))])



_vertexT_defs = Dict{Symbol, Tuple{Int,DataType}}([
    (:next => (0, QHvertexT)),
    (:previous => (1, QHvertexT)),
    (:point => (2, QHpointT)),
    (:neighbors => (3, QHsetT{QHfacetT})),
    (:id => (4, QHuint)),
    (:visitid => (5, QHuint)),
    (:seen => (6, QHflagT)),
    (:seen2 => (7, QHflagT)),
    (:deleted => (8, QHflagT)),
    (:delridge => (9, QHflagT)),
    (:newfacet => (10, QHflagT)),
    (:partitioned => (11, QHflagT))])


end # module
