# frozen_string_literal: true

# Pure-Ruby (no Jekyll dependency) logic for building a paragraph-by-paragraph
# bilingual "compare" view from an English markdown file and its Chinese
# translation.
#
# Design notes
# ------------
# * The English (`.md`) and Chinese (`.zh.md`) sources were translated
#   in-place, so their *structural* skeleton (figures, code fences, block
#   math, details blocks, headings) appears in the same order in both files.
#   Verification across all 13 chapters confirmed identical ordered figure
#   paths and >=0.98 structural-anchor sequence similarity everywhere.
# * We therefore split each file into blank-line-separated blocks (respecting
#   code/math/details spans that may contain blank lines) and merge at the
#   *source* level, before Jekyll renders anything.
# * Language-neutral blocks (code, block math, figures) are emitted ONCE;
#   figures additionally get a Chinese caption addendum when it differs.
# * Translated blocks (prose paragraphs, headings, details bodies) are paired
#   EN-then-ZH.
# * Structural blocks act as sync anchors. Between two anchors we pair prose
#   1:1; if a segment has unequal prose counts (the only real divergence seen,
#   e.g. training +5 / gpus -2 blocks), we pair what we can and then render the
#   leftover EN run followed by the leftover ZH run, so a local mismatch can
#   never poison the rest of the chapter.

module CompareCore
  FIGURE_RE = /\{%\s*include\s+figure\b/.freeze
  DETAILS_OPEN_RE = /\{%-?\s*details\b/.freeze
  DETAILS_CLOSE_RE = /\{%-?\s*enddetails\s*%}/.freeze
  HEADING_RE = /\A\s{0,3}\#{1,6}\s/.freeze

  module_function

  # Split raw markdown into an array of [text, kind] blocks.
  # kinds: :prose, :code, :math, :figure, :details
  def split_blocks(text)
    lines = text.gsub(/\r\n/, "\n").split("\n")
    blocks = []
    cur = []
    state = :normal # :normal | :code | :math | :details

    lines.each do |line|
      case state
      when :code
        cur << line
        if line.strip == "```"
          push_block(blocks, cur, :code)
          cur = []
          state = :normal
        end
      when :math
        cur << line
        if line.strip == "$$"
          push_block(blocks, cur, :math)
          cur = []
          state = :normal
        end
      when :details
        cur << line
        if line =~ DETAILS_CLOSE_RE
          push_block(blocks, cur, :details)
          cur = []
          state = :normal
        end
      else # :normal
        stripped = line.strip
        if stripped.empty?
          push_block(blocks, cur)
          cur = []
        elsif stripped.start_with?("```")
          push_block(blocks, cur)
          cur = [line]
          state = :code
        elsif stripped == "$$"
          push_block(blocks, cur)
          cur = [line]
          state = :math
        elsif line =~ DETAILS_OPEN_RE
          push_block(blocks, cur)
          cur = [line]
          state = :details
        else
          cur << line
        end
      end
    end
    push_block(blocks, cur)
    blocks
  end

  def push_block(blocks, cur, kind = nil)
    return if cur.empty?

    txt = cur.join("\n")
    kind ||= if txt.strip =~ /\A\{%\s*include\s+figure\b/
             :figure
           else
             :prose
           end
    blocks << [txt, kind]
  end

  # ---- classification helpers ----
  def once?(kind)
    %i[code math figure].include?(kind)
  end

  def pairable?(kind)
    %i[prose details].include?(kind)
  end

  # ---- interleave ----
  #
  # Language-neutral blocks (code/math/figure) are hard sync anchors. We walk
  # the two block lists segment by segment: between two anchors we pair the
  # pairable (prose/details) runs 1:1, and render any leftover run as a group
  # (EN group then ZH group) so a local prose-count mismatch can never shift
  # the anchors. At each anchor pair we emit the neutral block once (figures
  # get a bilingual caption addendum).
  def interleave(en_blocks, zh_blocks)
    out = []
    i = 0
    j = 0
    n = en_blocks.size
    m = zh_blocks.size

    while i < n && j < m
      a_e = next_anchor(en_blocks, i)
      a_z = next_anchor(zh_blocks, j)

      # Pairable segment between the current position and the next anchor.
      pair_runs(en_blocks[i...a_e], zh_blocks[j...a_z], out)

      i = a_e
      j = a_z
      break if i >= n || j >= m

      ek = en_blocks[i][1]
      zk = zh_blocks[j][1]
      if ek == zk
        out << (once?(ek) ? emit_once(ek, en_blocks[i][0], zh_blocks[j][0]) : emit_pair(en_blocks[i][0], zh_blocks[j][0]))
        i += 1
        j += 1
      else
        # Anchors of differing kind (defensive; verification says this won't
        # happen). Emit each alone so neither is dropped.
        out << emit_once(ek, en_blocks[i][0], "")
        out << emit_once(zk, "", zh_blocks[j][0])
        i += 1
        j += 1
      end
    end

    while i < n
      out << (once?(en_blocks[i][1]) ? emit_once(en_blocks[i][1], en_blocks[i][0], "") : wrap_en(en_blocks[i][0]))
      i += 1
    end
    while j < m
      out << (once?(zh_blocks[j][1]) ? emit_once(zh_blocks[j][1], "", zh_blocks[j][0]) : wrap_zh(zh_blocks[j][0]))
      j += 1
    end

    out.join("\n\n")
  end

  # Index of the next neutral anchor (code/math/figure), or blocks.size.
  def next_anchor(blocks, start)
    i = start
    while i < blocks.size
      return i if once?(blocks[i][1])

      i += 1
    end
    blocks.size
  end

  def pair_runs(seg_e, seg_z, out)
    k = [seg_e.size, seg_z.size].min
    k.times do |x|
      if seg_e[x][1] == :details
        out << emit_details_pair(seg_e[x][0], seg_z[x][0])
      else
        out << emit_pair(seg_e[x][0], seg_z[x][0])
      end
    end
    seg_e[k..].each { |b| out << wrap_en(b[0]) }
    seg_z[k..].each { |b| out << wrap_zh(b[0]) }
  end

  # A `{% details %}` block may contain a language-neutral figure include.
  # Emit such figures exactly once (with a bilingual caption when the ZH
  # caption differs) and strip them from both language copies, so we never
  # produce duplicate Distill figure ids.
  def emit_details_pair(en_text, zh_text)
    en_figs, en_body = extract_figures(en_text.to_s)
    zh_figs, zh_body = extract_figures(zh_text.to_s)

    seen = {}
    out = []
    (en_figs + zh_figs).each do |fig|
      path = figure_path(fig)
      next if seen[path]

      seen[path] = true
      zh_match = zh_figs.find { |z| figure_path(z) == path }
      out << emit_once(:figure, fig, zh_match || fig)
    end
    out << wrap_en(en_body.strip)
    out << wrap_zh(zh_body.strip)
    out.join("\n\n")
  end

  # Split text into [figure_include_lines, text_without_figures].
  def extract_figures(text)
    figs = []
    body = []
    text.each_line do |line|
      if line.strip =~ FIGURE_RE
        figs << line.strip
      else
        body << line
      end
    end
    [figs, body.join]
  end

  def figure_path(fig)
    m = fig.match(/path="([^"]+)"/)
    m ? m[1] : fig
  end

  # ---- emitters ----
  def emit_once(kind, en_text, zh_text)
    case kind
    when :code, :math
      en_text.to_s
    when :figure
      en_cap = caption_of(en_text)
      zh_cap = caption_of(zh_text)
      result = en_text.to_s
      if zh_cap && en_cap && zh_cap != en_cap
        result += "\n\n<div class=\"fig-caption-zh\" lang=\"zh\" markdown=\"1\">\n\n#{zh_cap}\n\n</div>"
      end
      result
    else
      en_text.to_s
    end
  end

  def emit_pair(en_text, zh_text)
    wrap_en(en_text.to_s) + "\n\n" + wrap_zh(zh_text.to_s)
  end

  def wrap_en(text)
    "<div class=\"src-en\" markdown=\"1\">\n\n#{text}\n\n</div>"
  end

  def wrap_zh(text)
    "<div class=\"src-zh\" lang=\"zh\" markdown=\"1\">\n\n#{text}\n\n</div>"
  end

  def caption_of(fig)
    return nil unless fig

    m = fig.match(/caption="(.*?)"/m)
    m ? m[1].strip : nil
  end
end
