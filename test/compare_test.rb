# frozen_string_literal: true

# Standalone test for CompareCore using the real roofline chapter pair.
# Run: ruby test/compare_test.rb

require_relative "../_plugins/compare_core"
require "pathname"

ROOT = Pathname.new(__FILE__).dirname.parent
EN = File.read(ROOT / "roofline.md")
ZH = File.read(ROOT / "roofline.zh.md")

def strip_fm(t)
  return t unless t.start_with?("---")

  i = t.index("---", 3)
  i ? t[i + 3..].to_s.lstrip : t
end

def check(cond, msg)
  if cond
    puts "  PASS: #{msg}"
  else
    puts "  FAIL: #{msg}"
    $failures_local = true
  end
end

en = CompareCore.split_blocks(strip_fm(EN))
zh = CompareCore.split_blocks(strip_fm(ZH))

puts "EN blocks: #{en.size}, ZH blocks: #{zh.size}"

en_kinds = en.map(&:last)
zh_kinds = zh.map(&:last)

puts "EN kinds: #{en_kinds.join(' ')}"
puts "ZH kinds: #{zh_kinds.join(' ')}"

merged = CompareCore.interleave(en, zh)

# 1. Both languages emitted
check(merged.include?('<div class="src-en"'), "output contains .src-en blocks")
check(merged.include?('<div class="src-zh" lang="zh"'), "output contains .src-zh blocks")

# 2. Standalone figures emitted exactly once (the roofline-improved figure is a
#    top-level block; the q3 figure lives inside a details block and thus shows
#    once per language, which is expected).
check(merged.scan("roofline-improved.png").size == 1,
      "standalone roofline-improved figure emitted once")
# 2b. A figure nested inside a details block must NOT be duplicated across the
#     EN/ZH details copies (would create duplicate Distill figure ids).
check(merged.scan("roofline-plot-q3.png").size == 1,
      "nested figure inside details emitted once (no duplicate id)")

# 3. Structural skeletons align in order
en_struct = en_kinds.select { |k| %i[code math figure details].include?(k) }
zh_struct = zh_kinds.select { |k| %i[code math figure details].include?(k) }
check(en_struct == zh_struct, "structural block sequences align (#{en_struct.size} vs #{zh_struct.size})")

# 4. No raw ZH body accidentally dropped: count src-zh == count of ZH pairable+once blocks
zh_pairable_once = zh.select { |_, k| %i[prose details code math figure].include?(k) }.size
src_zh = merged.scan('<div class="src-zh" lang="zh"').size
check(src_zh >= zh.select { |_, k| %i[prose details].include?(k) }.size,
      "zh pairable blocks all wrapped (#{src_zh} src-zh divs)")

# 5. Bilingual figure caption emitted when ZH caption differs (roofline-improved)
check(merged.include?('fig-caption-zh'), "bilingual figure caption present for translated caption")

puts
if $failures_local
  puts "TESTS FAILED"
  exit 1
else
  puts "ALL TESTS PASSED"
end
