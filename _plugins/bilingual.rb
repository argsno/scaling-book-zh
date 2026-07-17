# frozen_string_literal: true

# Bilingual collation plugin.
#
# For every English Distill chapter page `X.md`, if a parallel translation
# `X.zh.md` exists, this plugin injects the *rendered* Chinese article body
# after the English <d-article> content, wrapped in a
# `<div class="translation-zh" lang="zh">`. The result is an
# English-then-Chinese stacked collation view, with all Distill markup
# (figures, footnotes, math, citations) rendered correctly because each
# file is rendered by Jekyll normally and we only stitch the two bodies.
#
# Detection is by naming convention (X.md -> X.zh.md), so no front matter
# changes to the English sources are required.

require "jekyll"

module Jekyll
  module Bilingual
    def self.collate(site)
      site.pages.each do |en_page|
        # Skip the Chinese pages themselves.
        next if en_page.path.end_with?(".zh.md")
        next unless en_page.output.to_s.include?("<d-article")

        zh_path = en_page.path.sub(/\.md\z/, ".zh.md")
        next unless File.exist?(zh_path)

        zh_page = site.pages.find { |p| p.path == zh_path }
        next unless zh_page && zh_page.output && !zh_page.output.empty?

        match = zh_page.output.match(/<d-article[^>]*>(.*?)<\/d-article>/m)
        next unless match

        injected = "\n<div class=\"translation-zh\" lang=\"zh\">\n" \
                   "#{match[1]}\n</div>\n"
        en_page.output.sub!(%(</d-article>), "#{injected}</d-article>")
      end
    end
  end
end

Jekyll::Hooks.register(:site, :post_render) do |site|
  Jekyll::Bilingual.collate(site)
end
