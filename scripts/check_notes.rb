#!/usr/bin/env ruby
# frozen_string_literal: true

require "cgi"
require "pathname"
require "set"

ROOT = File.expand_path("..", __dir__)
ALLOWED_STATUSES = %w[stub draft reviewed].freeze
BANNED_PUBLIC_REFS = [
  "人工智能数学原理与算法-讲义.pdf",
  "26春期末重点.png"
].freeze

def relative(path)
  Pathname.new(path).relative_path_from(Pathname.new(ROOT)).to_s
end

def parse_frontmatter(content)
  lines = content.lines
  return { lines: [], body: content, aliases: [], status: nil } unless lines.first&.strip == "---"

  closing_offset = lines.drop(1).index { |line| line.strip == "---" }
  return nil unless closing_offset

  closing = closing_offset + 1
  frontmatter = lines[1...closing]
  aliases = []
  status = nil
  reading_aliases = false

  frontmatter.each do |line|
    if line.match?(/^aliases:\s*$/)
      reading_aliases = true
    elsif reading_aliases && (match = line.match(/^\s+-\s+(.+?)\s*$/))
      aliases << match[1].sub(/\A["']/, "").sub(/["']\z/, "")
    else
      reading_aliases = false unless line.match?(/^\s/)
      status = Regexp.last_match(1) if line =~ /^status:\s*(\S+)\s*$/
    end
  end

  {
    lines: frontmatter,
    body: lines.drop(closing + 1).join,
    aliases: aliases,
    status: status
  }
end

markdown_files = Dir.glob(File.join(ROOT, "**", "*.md"))
                    .reject { |path| path.include?("/.git/") }
                    .sort

errors = []
documents = {}
names = Hash.new { |hash, key| hash[key] = Set.new }

markdown_files.each do |path|
  content = File.read(path, encoding: "UTF-8")
  metadata = parse_frontmatter(content)
  rel = relative(path)

  unless metadata
    errors << "#{rel}: frontmatter 没有结束标记"
    next
  end

  documents[path] = { content: content, metadata: metadata }
  stem = File.basename(path, ".md")
  rel_without_ext = rel.sub(/\.md\z/, "")
  ([stem, rel_without_ext] + metadata[:aliases]).each { |name| names[name] << rel }

  if metadata[:body].strip.empty?
    errors << "#{rel}: 笔记正文为空"
  end

  if metadata[:status] && !ALLOWED_STATUSES.include?(metadata[:status])
    errors << "#{rel}: 未知 status '#{metadata[:status]}'"
  end

  math_fences = content.lines.count do |line|
    line.match?(/^\s*>?\s*\$\$\s*$/)
  end
  errors << "#{rel}: 展示公式的 $$ 没有成对出现" if math_fences.odd?

  content.lines.each_with_index do |line, index|
    errors << "#{rel}:#{index + 1}: 行尾有多余空白" if line.match?(/[ \t]+\n\z/)

    normalized = line.sub(/^\s*>\s?/, "").strip
    if normalized.include?("$$") && normalized != "$$"
      errors << "#{rel}:#{index + 1}: 展示公式应让 $$ 单独占行"
    end
  end

  BANNED_PUBLIC_REFS.each do |reference|
    errors << "#{rel}: 引用了未公开资料 #{reference}" if content.include?(reference)
  end
end

names.each do |name, paths|
  next unless paths.size > 1

  errors << "名称或 alias '#{name}' 指向多个文件: #{paths.to_a.sort.join(', ')}"
end

documents.each do |path, document|
  rel = relative(path)
  content = document[:content]

  # README 中包含用于说明语法的 `[[...]]` 示例，不把它当作真实双链。
  unless rel == "README.md"
    content.scan(/!?\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/) do |match|
      target = match[0].tr("\\", "/").strip
      basename = File.basename(target)
      source_relative = File.expand_path(target, File.dirname(path))
      root_relative = File.expand_path(target, ROOT)

      resolved = names.key?(target) || names.key?(basename) ||
                 File.exist?(source_relative) || File.exist?(root_relative)
      errors << "#{rel}: 无法解析双链 [[#{target}]]" unless resolved
    end
  end

  content.scan(/\]\(([^)]+)\)/) do |match|
    raw_target = match[0].strip
    next if raw_target.match?(/\A(?:https?:\/\/|mailto:|#)/)

    raw_target = raw_target[1...-1] if raw_target.start_with?("<") && raw_target.end_with?(">")
    target = CGI.unescape(raw_target.split("#", 2).first)
    next if target.empty?

    resolved = File.exist?(File.expand_path(target, File.dirname(path)))
    errors << "#{rel}: 无法解析 Markdown 链接 (#{raw_target})" unless resolved
  end
end

if errors.empty?
  puts "笔记检查通过：#{markdown_files.size} 篇 Markdown，名称唯一、链接和公式结构正常。"
  exit 0
end

warn "笔记检查失败（#{errors.size} 项）："
errors.each { |error| warn "- #{error}" }
exit 1
