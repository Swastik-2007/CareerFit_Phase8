"""Deterministic graph-route search and replay (not a full prerequisite plan)."""
import argparse
from collections import deque
import heapq
import json
import math
from pathlib import Path

from careerfit.growth import SkillGraph


def search(graph, start, goal, algorithm='bfs'):
    if algorithm not in ('bfs', 'dfs', 'ucs', 'astar'):
        raise ValueError('Unknown algorithm')
    if start not in graph or goal not in graph:
        raise ValueError('Start and goal must be graph skills')
    for node in graph:
        cost = graph.nodes[node]['estimated_hours']
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError('Learning costs must be finite and positive')
    # Reverse Dijkstra supplies an admissible exact route-cost lower bound.
    # Precomputation is disclosed separately from forward expansion counts.
    heuristic = {goal: 0}
    if algorithm == 'astar':
        queue = [(0, goal)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != heuristic[node]:
                continue
            for parent in sorted(graph.predecessors(node)):
                value = cost + graph.nodes[node]['estimated_hours']
                if value < heuristic.get(parent, math.inf):
                    heuristic[parent] = value
                    heapq.heappush(queue, (value, parent))
    pending = deque([(start, [start], 0)])
    heap = [(heuristic.get(start, 0) if algorithm == 'astar' else 0, 0, start, [start])]
    best = {start: 0}
    seen = set()
    expanded = []
    path, total = [], None
    while heap if algorithm in ('ucs', 'astar') else pending:
        if algorithm in ('ucs', 'astar'):
            _, cost, node, route = heapq.heappop(heap)
            if cost != best[node]:
                continue
        else:
            node, route, cost = pending.pop() if algorithm == 'dfs' else pending.popleft()
        if node in seen:
            continue
        seen.add(node)
        expanded.append(node)
        if node == goal:
            path, total = route, cost
            break
        neighbors = sorted(graph.successors(node), reverse=algorithm == 'dfs')
        for child in neighbors:
            value = cost + graph.nodes[child]['estimated_hours']
            if child in seen:
                continue
            if algorithm in ('ucs', 'astar'):
                if value < best.get(child, math.inf):
                    best[child] = value
                    priority = value + (heuristic.get(child, math.inf) if algorithm == 'astar' else 0)
                    heapq.heappush(heap, (priority, value, child, route + [child]))
            else:
                pending.append((child, route + [child], value))
    return dict(algorithm=algorithm, start=start, goal=goal, found=bool(path),
                path=path, route_hours=total, expanded=expanded, expanded_count=len(expanded),
                heuristic_precomputed_nodes=len(heuristic) if algorithm == 'astar' else 0,
                cost_policy='Start already known; charge each entered skill. A route does not cover all prerequisites.')


def visualize(graph, result, screenshot=None):
    import pygame
    import networkx as nx
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption('CareerFit Phase 7 - Search Replay')
    font = pygame.font.Font(None, 22)
    relevant = nx.ancestors(graph, result['goal']) | {result['goal'], result['start']}
    view = graph.subgraph(relevant)
    layers = list(nx.topological_generations(view))
    positions = {n: (90 + i * 1010 / max(1, len(layers)-1), 130 + j * 580 / max(1, len(layer)))
                 for i, layer in enumerate(layers) for j, n in enumerate(sorted(layer))}
    step, playing, running = 0, True, True
    clock = pygame.time.Clock()
    elapsed = 0
    while running:
        elapsed += clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_SPACE: playing = not playing
                if event.key == pygame.K_RIGHT: step = min(step+1, len(result['expanded']))
                if event.key == pygame.K_r: step = 0
        if playing and elapsed > 450:
            step = min(step+1, len(result['expanded']))
            elapsed = 0
        screen.fill((18, 24, 39))
        title = f"{result['algorithm'].upper()} | {result['start']} -> {result['goal']} | expanded {step}/{result['expanded_count']}"
        screen.blit(font.render(title, True, 'white'), (25, 20))
        screen.blit(font.render('Space: pause | Right: step | R: restart | Esc: close | Orange: visited | Green: final route', True, 'white'), (25, 50))
        screen.blit(font.render('Routes are demonstrations; use growth for complete prerequisite learning plans.', True, 'white'), (25, 76))
        for a, b in view.edges:
            pygame.draw.line(screen, (75, 90, 115), positions[a], positions[b], 2)
            ax, ay = positions[a]; bx, by = positions[b]
            angle = math.atan2(by-ay, bx-ax)
            tip = (bx-12*math.cos(angle), by-12*math.sin(angle))
            pygame.draw.polygon(screen, (130, 145, 170), [tip, (tip[0]-12*math.cos(angle-.4), tip[1]-12*math.sin(angle-.4)), (tip[0]-12*math.cos(angle+.4), tip[1]-12*math.sin(angle+.4))])
        for node, pos in positions.items():
            color = (65, 100, 160)
            if node in result['expanded'][:step]: color = (230, 160, 60)
            if step == len(result['expanded']) and node in result['path']: color = (65, 200, 135)
            pygame.draw.circle(screen, color, pos, 10)
            screen.blit(font.render(node, True, 'white'), (pos[0]-45, pos[1]+14))
        pygame.display.flip()
        if screenshot:
            pygame.image.save(screen, screenshot)
            running = False
    pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--algorithm', choices=['bfs', 'dfs', 'ucs', 'astar', 'all'], default='all')
    parser.add_argument('--output', default='output/search_results.json')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--screenshot', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        graph = SkillGraph().graph
        results = [search(graph, args.start, args.goal, a) for a in
                   (['bfs', 'dfs', 'ucs', 'astar'] if args.algorithm == 'all' else [args.algorithm])]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'careerfit_phase': 7, 'results': results}, indent=2), encoding='utf-8')
        print(f'Wrote {len(results)} search result(s) to {output}')
        if args.visualize:
            for result in results:
                visualize(graph, result, args.screenshot)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f'Search failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
