"""Simplified, deterministic rebalancing replay for the Oslo City Bike pilot."""

from collections import Counter
import heapq

import numpy as np


BIKES_PER_MOVE = 3
OUTSIDE = -1
NANOSECONDS_PER_HOUR = 60 * 60 * 1_000_000_000


def prepare_requests(trips, times, station_ids, start, stop):
    station_index = {sid: index for index, sid in enumerate(station_ids)}
    starts = trips["start_station_id"].map(station_index).fillna(OUTSIDE).astype(int)
    ends = trips["end_station_id"].map(station_index).fillna(OUTSIDE).astype(int)
    start_times = times["started_at"]
    end_times = times["ended_at"]
    mask = ((starts != OUTSIDE) | (ends != OUTSIDE)) & start_times.between(start, stop, inclusive="left")
    if (end_times[mask] <= start_times[mask]).any():
        raise ValueError("Replay requires each trip to end after it begins")
    requests = sorted(zip(start_times[mask].array.as_unit("ns").asi8,
                          end_times[mask].array.as_unit("ns").asi8,
                          starts[mask], ends[mask]))
    return requests


def replay(hours, active, stations, requests, warmup, policy, forecast=None):
    capacities = np.array([s["capacity"] for s in stations], dtype=int)
    stock = capacities // 2
    initial_stock = int(stock.sum())
    waiting = np.zeros_like(stock)
    transfers = Counter()
    returns = []
    next_request = 0
    history = np.zeros((len(warmup) + len(hours), len(stations), 2))
    history[:len(warmup)] = warmup

    def drain(station):
        available_docks = capacities[station] - stock[station]
        amount = min(available_docks, waiting[station])
        stock[station] += amount
        waiting[station] -= amount
        return amount

    def transfer(t):
        target = capacities / 2
        if forecast is not None:
            predicted_flows = forecast(t, history)
            expected_net_checkouts = predicted_flows[:, 0] - predicted_flows[:, 1]
            target = np.clip(target + expected_net_checkouts, 0, capacities)
        surplus = stock - target
        donor = int(np.argmax(surplus))
        recipient = int(np.argmin(surplus))
        amount = int(min(BIKES_PER_MOVE, surplus[donor], -surplus[recipient]))
        if amount <= 0:
            return
        stock[donor] -= amount
        stock[recipient] += amount
        history[len(warmup) + t, donor, 1] += drain(donor)
        transfers["moves"] += 1
        transfers["bikes_moved"] += amount

    for t, (hour, is_active) in enumerate(zip(hours, active)):
        current = history[len(warmup) + t]
        if policy != "no_transfer" and is_active:
            transfer(t)
        boundary = hour.value + NANOSECONDS_PER_HOUR
        while True:
            next_departure = requests[next_request][0] if next_request < len(requests) else float("inf")
            next_return = returns[0][0] if returns else float("inf")
            if min(next_departure, next_return) >= boundary:
                break
            if next_return <= next_departure:
                _, station = heapq.heappop(returns)
                transfers["return_attempts"] += 1
                if stock[station] < capacities[station]:
                    stock[station] += 1
                    current[station, 1] += 1
                else:
                    transfers["blocked_returns"] += 1
                    waiting[station] += 1
                continue
            _, end, origin, destination = requests[next_request]
            next_request += 1
            if origin == OUTSIDE:
                transfers["bikes_entered"] += 1
            else:
                transfers["checkout_attempts"] += 1
                if stock[origin] == 0:
                    transfers["blocked_checkouts"] += 1
                    continue
                stock[origin] -= 1
                current[origin, 0] += 1
                current[origin, 1] += drain(origin)
            if destination == OUTSIDE:
                transfers["bikes_left"] += 1
            else:
                heapq.heappush(returns, (end, destination))

        total = int(stock.sum() + waiting.sum()) + len(returns)
        expected = initial_stock + transfers["bikes_entered"] - transfers["bikes_left"]
        if total != expected or (stock < 0).any() or (stock > capacities).any():
            raise AssertionError("Replay violated bike conservation or a station capacity")

    counters = ("checkout_attempts", "blocked_checkouts", "return_attempts", "blocked_returns",
                "moves", "bikes_moved", "bikes_entered", "bikes_left")
    result = {key: transfers[key] for key in counters}
    result["end_waiting_to_dock"] = int(waiting.sum())
    result["end_in_transit"] = len(returns)
    return result
