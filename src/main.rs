use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use plotters::coord::Shift;
use plotters::prelude::*;

/// A simple 2D coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct Point {
    x: i32,
    y: i32,
}

/// A 2D grid storing obstacles (true) and free space (false).
struct Grid {
    width: usize,
    height: usize,
    cells: Vec<Vec<bool>>,
}

impl Grid {
    /// Create a new grid with `width` columns and `height` rows, all free.
    fn new(width: usize, height: usize) -> Self {
        let cells = vec![vec![false; width]; height];
        Grid {
            width,
            height,
            cells,
        }
    }

    /// Check if a point is within the grid boundaries.
    fn in_bounds(&self, p: Point) -> bool {
        p.x >= 0 && p.x < self.width as i32 && p.y >= 0 && p.y < self.height as i32
    }

    /// Check if the cell is an obstacle.
    fn is_obstacle(&self, p: Point) -> bool {
        self.cells[p.y as usize][p.x as usize]
    }

    /// Set a cell as an obstacle.
    fn set_obstacle(&mut self, p: Point) {
        if self.in_bounds(p) {
            self.cells[p.y as usize][p.x as usize] = true;
        }
    }

    /// Get valid (non-obstacle) neighbors in four directions.
    fn neighbors(&self, p: Point) -> Vec<Point> {
        let deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut result = Vec::new();
        for (dx, dy) in deltas {
            let nx = p.x + dx;
            let ny = p.y + dy;
            let neighbor = Point { x: nx, y: ny };
            if self.in_bounds(neighbor) && !self.is_obstacle(neighbor) {
                result.push(neighbor);
            }
        }
        result
    }
}

/// Node for Dijkstra and A* in a BinaryHeap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Node {
    cost: i32,
    point: Point,
}

/// Implement `Ord` so it becomes a min-heap (by inverting comparison).
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // We'll do cost ascending => invert
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reconstruct path from a parent map.
fn reconstruct_path(
    parents: &HashMap<Point, Option<Point>>,
    start: Point,
    goal: Point,
) -> Vec<Point> {
    let mut path = Vec::new();
    let mut current = goal;
    while current != start {
        path.push(current);
        if let Some(p) = parents[&current] {
            current = p;
        } else {
            break;
        }
    }
    path.push(start);
    path.reverse();
    path
}

/// Manhattan distance heuristic for A*.
fn heuristic(a: Point, b: Point) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

/// Clear the terminal using ANSI escape codes.
fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
}

/// Print grid state in ASCII:
/// - 'S' for start
/// - 'G' for goal
/// - '#' for obstacle
/// - 'F' for frontier
/// - 'V' for visited
/// - '.' for free
fn print_grid_state(
    grid: &Grid,
    start: Point,
    goal: Point,
    visited: &HashSet<Point>,
    frontier: &HashSet<Point>,
) {
    clear_screen();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let p = Point {
                x: x as i32,
                y: y as i32,
            };
            if p == start {
                print!("S");
            } else if p == goal {
                print!("G");
            } else if grid.is_obstacle(p) {
                print!("#");
            } else if frontier.contains(&p) {
                print!("F");
            } else if visited.contains(&p) {
                print!("V");
            } else {
                print!(".");
            }
        }
        println!();
    }
    println!();
}

/// Plot the grid to the given drawing area, marking visited, frontier, etc.
fn draw_grid_on_area<DB>(
    area: &DrawingArea<DB, Shift>,
    grid: &Grid,
    visited: &HashSet<Point>,
    frontier: &HashSet<Point>,
    path: Option<&[Point]>,
    start: Point,
    goal: Point,
) -> Result<(), Box<dyn std::error::Error + 'static>>
where
    DB: DrawingBackend + 'static,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    area.fill(&WHITE)?;

    let cell_size = 20; // smaller cell to accommodate bigger grid in the image
    let path_set: HashSet<Point> = path.unwrap_or_default().iter().copied().collect();

    for y in 0..grid.height {
        for x in 0..grid.width {
            let p = Point { x: x as i32, y: y as i32 };
            let x0 = (x * cell_size) as i32;
            let y0 = (y * cell_size) as i32;
            let x1 = x0 + cell_size as i32 - 1;
            let y1 = y0 + cell_size as i32 - 1;

            // Make sure all branches return `ShapeStyle`
            let color = if p == start {
                RGBColor(0, 200, 0).filled() // green
            } else if p == goal {
                RGBColor(200, 0, 0).filled() // red
            } else if grid.is_obstacle(p) {
                BLACK.filled()
            } else if path_set.contains(&p) {
                RGBColor(255, 215, 0).filled() // gold
            } else if frontier.contains(&p) {
                RGBColor(0, 0, 255).filled() // blue
            } else if visited.contains(&p) {
                RGBColor(180, 180, 180).filled() // light gray
            } else {
                WHITE.filled()
            };

            area.draw(&Rectangle::new(
                [(x0, y0), (x1, y1)],
                color,
            ))?;
        }
    }

    Ok(())
}

/// A helper function to overlay some text onto the drawing.
fn draw_text<DB>(
    area: &DrawingArea<DB, Shift>,
    text: &str,
    offset_y: i32,
) -> Result<(), Box<dyn std::error::Error + 'static>>
where
    DB: DrawingBackend + 'static,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let style = ("sans-serif", 16).into_font().color(&BLACK);
    area.draw(&Text::new(text, (10, offset_y), style))?;
    Ok(())
}

/// BFS with both ASCII printing & plotters callback.
fn bfs<F>(
    grid: &Grid,
    start: Point,
    goal: Point,
    mut step_cb: F,
) -> Option<Vec<Point>>
where
    // BFS calls `step_cb(iter, visited, frontier_queue)`
    F: FnMut(usize, &HashSet<Point>, &VecDeque<Point>),
{
    if !grid.in_bounds(start) || !grid.in_bounds(goal) {
        return None;
    }
    if grid.is_obstacle(start) || grid.is_obstacle(goal) {
        return None;
    }

    let mut visited = HashSet::new();
    visited.insert(start);
    let mut queue = VecDeque::new();
    queue.push_back(start);

    let mut parent_map = HashMap::new();
    parent_map.insert(start, None);

    let mut iteration = 0;

    while let Some(current) = queue.pop_front() {
        step_cb(iteration, &visited, &queue);
        iteration += 1;

        if current == goal {
            // reconstruct path
            return Some(reconstruct_path(&parent_map, start, goal));
        }

        for nb in grid.neighbors(current) {
            if !visited.contains(&nb) {
                visited.insert(nb);
                parent_map.insert(nb, Some(current));
                queue.push_back(nb);
            }
        }
    }

    None
}

/// Dijkstra with ASCII printing & plotters callback.
fn dijkstra<F>(
    grid: &Grid,
    start: Point,
    goal: Point,
    mut step_cb: F,
) -> Option<Vec<Point>>
where
    // Dijkstra calls `step_cb(iter, visited, frontier_nodes)`
    F: FnMut(usize, &HashSet<Point>, &Vec<Node>),
{
    if !grid.in_bounds(start) || !grid.in_bounds(goal) {
        return None;
    }
    if grid.is_obstacle(start) || grid.is_obstacle(goal) {
        return None;
    }

    let mut dist = HashMap::new();
    let mut parents = HashMap::new();
    let mut visited = HashSet::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start, 0);
    parents.insert(start, None);
    heap.push(Node { cost: 0, point: start });

    let mut iteration = 0;

    while let Some(Node { cost, point: current }) = heap.pop() {
        let mut frontier_vec: Vec<Node> = heap.clone().into_vec();
        frontier_vec.sort_by_key(|n| n.cost);

        step_cb(iteration, &visited, &frontier_vec);
        iteration += 1;

        if current == goal {
            return Some(reconstruct_path(&parents, start, goal));
        }

        if cost > *dist.get(&current).unwrap_or(&i32::MAX) {
            continue;
        }

        visited.insert(current);

        for nb in grid.neighbors(current) {
            let new_cost = cost + 1;
            if new_cost < *dist.get(&nb).unwrap_or(&i32::MAX) {
                dist.insert(nb, new_cost);
                parents.insert(nb, Some(current));
                heap.push(Node { cost: new_cost, point: nb });
            }
        }
    }

    None
}

/// A* with ASCII printing & plotters callback.
fn astar<F>(
    grid: &Grid,
    start: Point,
    goal: Point,
    mut step_cb: F,
) -> Option<Vec<Point>>
where
    // A* calls `step_cb(iter, visited, frontier_nodes)`
    F: FnMut(usize, &HashSet<Point>, &Vec<Node>),
{
    if !grid.in_bounds(start) || !grid.in_bounds(goal) {
        return None;
    }
    if grid.is_obstacle(start) || grid.is_obstacle(goal) {
        return None;
    }

    let mut g_cost = HashMap::new();
    let mut parents = HashMap::new();
    let mut visited = HashSet::new();
    let mut heap = BinaryHeap::new();

    g_cost.insert(start, 0);
    let start_h = heuristic(start, goal);
    heap.push(Node {
        cost: start_h,
        point: start,
    });
    parents.insert(start, None);

    let mut iteration = 0;

    while let Some(Node { cost: _, point: current }) = heap.pop() {
        let mut frontier_vec: Vec<Node> = heap.clone().into_vec();
        frontier_vec.sort_by_key(|n| n.cost);

        step_cb(iteration, &visited, &frontier_vec);
        iteration += 1;

        if current == goal {
            return Some(reconstruct_path(&parents, start, goal));
        }

        visited.insert(current);

        for nb in grid.neighbors(current) {
            let tentative_g = g_cost[&current] + 1;
            if tentative_g < *g_cost.get(&nb).unwrap_or(&i32::MAX) {
                g_cost.insert(nb, tentative_g);
                let f = tentative_g + heuristic(nb, goal);
                parents.insert(nb, Some(current));
                heap.push(Node { cost: f, point: nb });
            }
        }
    }

    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make a bigger grid, e.g. 20 x 15
    let mut grid = Grid::new(20, 15);

    // We place some obstacles in more complicated patterns
    // E.g.: a horizontal wall in row=5, col=2..17
    for x in 2..18 {
        grid.set_obstacle(Point { x: x as i32, y: 5 });
    }
    // A vertical wall in col=10, row=6..13
    for y in 6..14 {
        grid.set_obstacle(Point { x: 10, y: y as i32 });
    }
    // A few random scattered blocks near bottom
    for x in 6..9 {
        for y in 12..15 {
            if (x + y) % 2 == 0 {
                grid.set_obstacle(Point { x, y });
            }
        }
    }

    let start = Point { x: 0, y: 0 };
    let goal = Point { x: 19, y: 14 };

    fs::create_dir_all("./output")?;

    // We'll produce BFS.gif, dijkstra.gif, astar.gif with ~150ms per frame
    let cell_size = 20;
    let width_px = grid.width * cell_size;
    let height_px = grid.height * cell_size;
    let frame_delay_ms = 150;

    // ---------------------------------------
    // BFS
    // ---------------------------------------
    let backend_bfs = BitMapBackend::gif(
        "./output/bfs.gif",
        (width_px as u32, height_px as u32),
        frame_delay_ms,
    )?;
    let root_bfs = backend_bfs.into_drawing_area();

    let bfs_path = bfs(&grid, start, goal, |iter, visited, queue| {
        // 1) ASCII printing
        let frontier_set: HashSet<Point> = queue.iter().copied().collect();
        print_grid_state(&grid, start, goal, visited, &frontier_set);
        println!("BFS Iteration: {}, visited={}, frontier={}", iter, visited.len(), queue.len());

        thread::sleep(Duration::from_millis(150));

        // 2) Plotters drawing
        draw_grid_on_area(&root_bfs, &grid, visited, &frontier_set, None, start, goal)
            .expect("draw BFS error");

        let label = format!(
            "BFS iter={} visited={} frontier={}",
            iter, visited.len(), queue.len()
        );
        draw_text(&root_bfs, &label, 20).unwrap();

        root_bfs.present().unwrap();
    });

    if let Some(path) = bfs_path {
        println!("BFS found path, length={}", path.len());
        // final ASCII
        {
            let visited = HashSet::new();
            let frontier = HashSet::new();
            print_grid_state(&grid, start, goal, &visited, &frontier);
            println!("Final BFS Path length={}", path.len());
        }
        // final GIF frame
        draw_grid_on_area(&root_bfs, &grid, &HashSet::new(), &HashSet::new(), Some(&path), start, goal)?;
        let label = format!("BFS done!\nPath length={}", path.len());
        draw_text(&root_bfs, &label, 20)?;
        root_bfs.present()?;
    } else {
        println!("BFS could not find a path!");
    }

    // ---------------------------------------
    // Dijkstra
    // ---------------------------------------
    let backend_dij = BitMapBackend::gif(
        "./output/dijkstra.gif",
        (width_px as u32, height_px as u32),
        frame_delay_ms,
    )?;
    let root_dij = backend_dij.into_drawing_area();

    let dijkstra_path = dijkstra(&grid, start, goal, |iter, visited, frontier_nodes| {
        // 1) ASCII
        let frontier_set: HashSet<Point> = frontier_nodes.iter().map(|node| node.point).collect();
        print_grid_state(&grid, start, goal, visited, &frontier_set);
        println!(
            "Dijkstra Iter={}, visited={}, frontier={}",
            iter,
            visited.len(),
            frontier_nodes.len()
        );
        thread::sleep(Duration::from_millis(150));

        // 2) Plot
        draw_grid_on_area(&root_dij, &grid, visited, &frontier_set, None, start, goal)
            .expect("draw Dijkstra error");

        let label = format!(
            "Dijkstra iter={} visited={} frontier={}",
            iter, visited.len(), frontier_nodes.len()
        );
        draw_text(&root_dij, &label, 20).unwrap();

        root_dij.present().unwrap();
    });

    if let Some(path) = dijkstra_path {
        println!("Dijkstra found path, length={}", path.len());
        // final ASCII
        {
            let visited = HashSet::new();
            let frontier = HashSet::new();
            print_grid_state(&grid, start, goal, &visited, &frontier);
            println!("Final Dijkstra Path length={}", path.len());
        }
        // final GIF frame
        draw_grid_on_area(&root_dij, &grid, &HashSet::new(), &HashSet::new(), Some(&path), start, goal)?;
        let label = format!("Dijkstra done!\nPath length={}", path.len());
        draw_text(&root_dij, &label, 20)?;
        root_dij.present()?;
    } else {
        println!("Dijkstra could not find a path!");
    }

    // ---------------------------------------
    // A*
    // ---------------------------------------
    let backend_astar = BitMapBackend::gif(
        "./output/astar.gif",
        (width_px as u32, height_px as u32),
        frame_delay_ms,
    )?;
    let root_astar = backend_astar.into_drawing_area();

    let astar_path = astar(&grid, start, goal, |iter, visited, frontier_nodes| {
        let frontier_set: HashSet<Point> = frontier_nodes.iter().map(|node| node.point).collect();
        // ASCII
        print_grid_state(&grid, start, goal, visited, &frontier_set);
        println!(
            "A* Iter={}, visited={}, frontier={}",
            iter,
            visited.len(),
            frontier_nodes.len()
        );
        thread::sleep(Duration::from_millis(150));

        // Plot
        draw_grid_on_area(&root_astar, &grid, visited, &frontier_set, None, start, goal)
            .expect("draw A* error");

        let label = format!(
            "A* iter={} visited={} frontier={}",
            iter, visited.len(), frontier_nodes.len()
        );
        draw_text(&root_astar, &label, 20).unwrap();

        root_astar.present().unwrap();
    });

    if let Some(path) = astar_path {
        println!("A* found path, length={}", path.len());
        // final ASCII
        {
            let visited = HashSet::new();
            let frontier = HashSet::new();
            print_grid_state(&grid, start, goal, &visited, &frontier);
            println!("Final A* Path length={}", path.len());
        }
        // final GIF
        draw_grid_on_area(&root_astar, &grid, &HashSet::new(), &HashSet::new(), Some(&path), start, goal)?;
        let label = format!("A* done!\nPath length={}", path.len());
        draw_text(&root_astar, &label, 20)?;
        root_astar.present()?;
    } else {
        println!("A* could not find a path!");
    }

    println!("All done. Check `./output` for BFS/Dijkstra/A* GIFs.");
    Ok(())
}