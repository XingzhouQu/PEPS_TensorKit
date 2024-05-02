"""
Convention: s1 and s2 are spaces to be exchanged.

Return space：(s1, s2) ← (s1, s2)
"""
function swap_gate(s1::T, s2::T) where T <: ElementarySpace
    tmp1 = id(s1)
    tmp2 = id(s2)
    tmp = tmp1 ⊗ tmp2
    for blk in blocks(tmp)
        for sct in blk[1].sectors
            if sct == Irrep[ℤ₂](1)
                block(tmp, blk[1]) .*= -1
                continue
            end
        end
    end
    return tmp
end

# 这样做也行，但是似乎慢一点，后面对大的张量可以再测试一下
# function swap_gate2(s1::T, s2::T) where T <: ElementarySpace
#     # convention: s1 and s2 are domain spaces.
#     tmp1 = id(s1)
#     tmp2 = id(s2)
#     for blk in blocks(tmp1)
#         for sct in blk[1].sectors
#             if sct == Irrep[ℤ₂](1)
#                 block(tmp1, blk[1]) .*= -1
#                 continue
#             end
#         end
#     end
#     for blk in blocks(tmp2)
#         for sct in blk[1].sectors
#             if sct == Irrep[ℤ₂](1)
#                 block(tmp2, blk[1]) .*= -1
#                 continue
#             end
#         end
#     end
#     return tmp1 ⊗ tmp2
# end
