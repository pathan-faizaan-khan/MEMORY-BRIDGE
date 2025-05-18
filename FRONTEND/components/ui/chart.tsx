"use client"

import type React from "react"

import { Slot } from "@radix-ui/react-slot"
import type { ChartTooltipProps } from "recharts/types/chart/generateCategoricalChart"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { cn } from "@/lib/utils"

// Chart wrapper component
interface ChartProps extends React.HTMLAttributes<HTMLDivElement> {
  asChild?: boolean
}

const Chart = ({ asChild = false, className, ...props }: ChartProps) => {
  const Comp = asChild ? Slot : "div"
  return <Comp className={cn("w-full", className)} {...props} />
}

// Chart tooltip component
const ChartTooltip = ({ className, ...props }: React.ComponentProps<typeof Tooltip>) => (
  <Tooltip
    cursor={false}
    content={({ active, payload }) => {
      if (active && payload && payload.length) {
        return (
          <div className="rounded-lg border bg-background p-2 shadow-sm">
            <div className="grid grid-cols-2 gap-2">
              {payload.map((entry) => (
                <div key={entry.name} className="flex flex-col">
                  <span className="text-[0.70rem] uppercase text-muted-foreground">{entry.name}</span>
                  <span className="font-bold text-muted-foreground">{entry.value}</span>
                </div>
              ))}
            </div>
          </div>
        )
      }

      return null
    }}
    {...props}
  />
)

// Custom tooltip for pie charts
const PieChartTooltip = ({ active, payload }: ChartTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg border bg-background p-2 shadow-sm">
        <div className="flex flex-col">
          <span className="text-[0.70rem] uppercase text-muted-foreground">{payload[0].name}</span>
          <span className="font-bold text-muted-foreground">{payload[0].value}</span>
          <span className="text-xs text-muted-foreground">
            {payload[0].payload.percent ? `${(payload[0].payload.percent * 100).toFixed(1)}%` : ""}
          </span>
        </div>
      </div>
    )
  }

  return null
}

// Chart components
const ChartContent = ResponsiveContainer
const ChartPie = Pie
const ChartLine = Line
const ChartBar = Bar
const ChartXAxis = XAxis
const ChartYAxis = YAxis
const ChartLegend = Legend
const ChartGrid = CartesianGrid
const ChartCell = Cell
const ChartRadialBar = RadialBar

// Chart container components
const LineChartContainer = LineChart
const BarChartContainer = BarChart
const PieChartContainer = PieChart
const RadialBarChartContainer = RadialBarChart

export {
  Chart,
  ChartContent,
  ChartTooltip,
  PieChartTooltip,
  ChartPie,
  ChartLine,
  ChartBar,
  ChartXAxis,
  ChartYAxis,
  ChartLegend,
  ChartGrid,
  ChartCell,
  ChartRadialBar,
  LineChartContainer,
  BarChartContainer,
  PieChartContainer,
  RadialBarChartContainer,
}
