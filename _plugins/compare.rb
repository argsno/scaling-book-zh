# frozen_string_literal: true

# Bilingual "compare" generator.
#
# For every English Distill chapter page `X.md` that has a parallel translation
# `X.zh.md`, this generator merges the two source files block-by-block (see
# CompareCore) and emits a new page at `/X-compare/` rendered with the same
# `distill` layout. The English page itself is left untouched (EN-only); the
# whole-article stacking previously done by bilingual.rb has been retired.
#
# Detection is by naming convention (X.md -> X.zh.md), so no front matter
# changes to the English sources are required. The EN front matter is copied
# onto the compare page; only layout/permalink/autogen/toc/title are overridden.

require "jekyll"
require_relative "compare_core"

module Jekyll
  class CompareGenerator < Jekyll::Generator
    priority :low

    def generate(site)
      targets = site.pages.select do |page|
        next false unless page.path.end_with?(".md")
        next false if page.path.end_with?(".zh.md")
        next false if page.data["compare_skip"]

        zh_path = page.path.sub(/\.md\z/, ".zh.md")
        File.exist?(File.join(site.source, zh_path))
      end

      targets.each do |en_page|
        base = File.basename(en_page.path).sub(/\.md\z/, "")
        permalink = "/#{base}-compare/"

        # Guard against duplicates (e.g. re-runs / already-created pages).
        next if site.pages.any? { |p| p.data["permalink"].to_s == permalink }

        zh_path = en_page.path.sub(/\.md\z/, ".zh.md")
        zh_page = site.pages.find { |p| p.path == zh_path }
        next unless zh_page

        merged = CompareCore.interleave(
          CompareCore.split_blocks(en_page.content.to_s),
          CompareCore.split_blocks(zh_page.content.to_s)
        )

        page = Jekyll::Page.new(site, site.source, "", "#{base}-compare.md")
        page.data.merge!(en_page.data)
        page.data["layout"] = "distill"
        page.data["permalink"] = permalink
        page.data["autogen"] = true
        page.data["toc"] = false
        page.data["title"] = "#{en_page.data['title']} · 对照"
        page.content = merged

        site.pages << page
      end
    end
  end
end
