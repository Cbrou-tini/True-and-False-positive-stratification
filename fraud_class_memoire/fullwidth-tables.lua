local CHAR_WIDTH_CM = 0.19
local LINEWIDTH_CM = 16.0
local MAX_MIN_FRAC = 0.40

local function cell_latex(cell)
  local doc = pandoc.Pandoc(cell.contents)
  local latex = pandoc.write(doc, "latex")
  latex = latex:gsub("%s+$", "")
  latex = latex:gsub("\n+", " ")
  return latex
end

local function longest_word(text)
  local best = 0
  for word in text:gmatch("%S+") do
    if #word > best then best = #word end
  end
  return best
end

function Table(tbl)
  local n = #tbl.colspecs
  if n == 0 then return tbl end

  local maxlen = {}
  local maxword = {}
  for i = 1, n do
    maxlen[i] = 1
    maxword[i] = 1
  end

  local function scan_row(row)
    for i, cell in ipairs(row.cells) do
      if i <= n then
        local text = pandoc.utils.stringify(cell.contents)
        local len = #text
        if len > maxlen[i] then maxlen[i] = len end
        local w = longest_word(text)
        if w > maxword[i] then maxword[i] = w end
      end
    end
  end

  for _, row in ipairs(tbl.head.rows) do scan_row(row) end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do scan_row(row) end
  end

  -- proportional starting point, based on typical (total) cell length
  local total = 0
  for i = 1, n do total = total + maxlen[i] end
  local propW = {}
  for i = 1, n do propW[i] = maxlen[i] / total end

  -- minimum width needed so the longest unbreakable word/token doesn't overflow
  local minFrac = {}
  for i = 1, n do
    local f = (maxword[i] + 2) * CHAR_WIDTH_CM / LINEWIDTH_CM
    if f > MAX_MIN_FRAC then f = MAX_MIN_FRAC end
    minFrac[i] = f
  end

  -- fix columns whose proportional share is below their minimum
  local fixed = {}
  local fixedTotal = 0
  for i = 1, n do
    if propW[i] < minFrac[i] then
      fixed[i] = minFrac[i]
      fixedTotal = fixedTotal + minFrac[i]
    end
  end

  local remaining = 1.0 - fixedTotal
  if remaining < 0.05 then remaining = 0.05 end

  local freeTotal = 0
  for i = 1, n do
    if not fixed[i] then freeTotal = freeTotal + propW[i] end
  end

  local widths = {}
  for i = 1, n do
    if fixed[i] then
      widths[i] = fixed[i]
    elseif freeTotal > 0 then
      widths[i] = (propW[i] / freeTotal) * remaining
    else
      widths[i] = remaining / n
    end
  end

  local sum = 0
  for i = 1, n do sum = sum + widths[i] end
  for i = 1, n do widths[i] = widths[i] / sum end

  -- (n-1) internal column boundaries each consume 2\tabcolsep of horizontal
  -- space that is not part of any column's specified width; that width must
  -- be subtracted up front or the rendered table overflows \linewidth.
  local gaps = n - 1
  local colspec = "@{}"
  for i = 1, n do
    colspec = colspec
      .. string.format(">{\\raggedright\\arraybackslash}p{(\\linewidth - %d\\tabcolsep) * \\real{%.4f}}", 2 * gaps, widths[i])
  end
  colspec = colspec .. "@{}"

  local function row_to_latex(row)
    local cells = {}
    for i, cell in ipairs(row.cells) do
      if i <= n then
        cells[i] = cell_latex(cell)
      end
    end
    return table.concat(cells, " & ") .. " \\\\"
  end

  local header_lines = {}
  for _, row in ipairs(tbl.head.rows) do
    header_lines[#header_lines + 1] = row_to_latex(row)
  end

  local body_lines = {}
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      body_lines[#body_lines + 1] = row_to_latex(row)
    end
  end

  local parts = {}
  parts[#parts + 1] = "\\begin{minipage}{\\linewidth}"
  parts[#parts + 1] = "\\footnotesize"
  parts[#parts + 1] = "\\begin{tabular}{" .. colspec .. "}"
  parts[#parts + 1] = "\\toprule"
  for _, l in ipairs(header_lines) do parts[#parts + 1] = l end
  parts[#parts + 1] = "\\midrule"
  for _, l in ipairs(body_lines) do parts[#parts + 1] = l end
  parts[#parts + 1] = "\\bottomrule"
  parts[#parts + 1] = "\\end{tabular}"
  parts[#parts + 1] = "\\end{minipage}"

  return pandoc.RawBlock("latex", table.concat(parts, "\n"))
end
