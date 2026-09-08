"""Fake KataGo analysis engine for adapter tests. Reads queries from stdin, writes responses.

Behaviour switches via environment variables:
  FAKE_TEACHER_SHUFFLE=1     emit turns of a query in reversed order
  FAKE_TEACHER_DUPLICATE=1   emit every response twice
  FAKE_TEACHER_DROP_TURN=N   never answer turn N of any query (first attempt only if RETRY_OK)
  FAKE_TEACHER_DROP_ONCE=1   with DROP_TURN: only drop on query ids ending with "-1" (first attempt)
  FAKE_TEACHER_SLEEP=S       sleep S seconds before answering each query
  FAKE_TEACHER_PERSPECTIVE=black|white|sidetomove   perspective of emitted values (default sidetomove)
  FAKE_TEACHER_ERROR_ID=X    reply {"error":...} for query id containing X
  FAKE_TEACHER_ILLEGAL=1     include visits on an illegal move (the last move played)
The fake evaluates: winrate(to-move) = 0.5 + 0.01*turn, scoreLead(to-move) = turn/10,
ownership(to-move) = +1 on the last black move point if black to move else -1 ... (deterministic).
"""
import json
import os
import sys
import time

GTP = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def idx_to_gtp(i, S):
    if i == S * S:
        return "pass"
    return f"{GTP[i % S]}{S - i // S}"


def main():
    persp = os.environ.get("FAKE_TEACHER_PERSPECTIVE", "sidetomove")
    for line in sys.stdin:
        q = json.loads(line)
        if os.environ.get("FAKE_TEACHER_SLEEP"):
            time.sleep(float(os.environ["FAKE_TEACHER_SLEEP"]))
        if os.environ.get("FAKE_TEACHER_ERROR_ID") and os.environ["FAKE_TEACHER_ERROR_ID"] in q["id"]:
            print(json.dumps({"id": q["id"], "error": "fake error", "field": "rules"}), flush=True)
            continue
        S = q["boardXSize"]
        turns = list(q["analyzeTurns"])
        if os.environ.get("FAKE_TEACHER_SHUFFLE"):
            turns = turns[::-1]
        for t in turns:
            drop = os.environ.get("FAKE_TEACHER_DROP_TURN")
            if drop is not None and int(drop) == t:
                if not os.environ.get("FAKE_TEACHER_DROP_ONCE") or q["id"].endswith("-1"):
                    continue
            moves = q["moves"][:t]
            to_move = "B" if (len(moves) % 2 == 0) == (q["initialPlayer"] == "B") else "W"
            win_tm = 0.5 + 0.01 * t
            lead_tm = t / 10.0
            own_tm = [0.0] * (S * S)
            own_tm[0] = 0.75  # to-move owns top-left, by definition of the fake
            played = {(S - int(m[1:])) * S + GTP.index(m[0]) for _, m in moves if m.lower() != "pass"}
            infos = []
            v = 50
            for i in range(S * S + 1):
                if i in played:
                    continue
                infos.append({"move": idx_to_gtp(i, S), "visits": v, "winrate": win_tm, "scoreLead": lead_tm, "order": len(infos)})
                v = max(1, v - 7)
                if len(infos) >= 8:
                    break
            if os.environ.get("FAKE_TEACHER_ILLEGAL") and moves:
                last = moves[-1][1]
                infos.append({"move": last, "visits": 3, "winrate": win_tm, "scoreLead": lead_tm, "order": len(infos)})
            sign = 1.0 if persp == "sidetomove" or persp[0].upper() == to_move else -1.0
            win = win_tm if sign > 0 else 1 - win_tm
            resp = {"id": q["id"], "turnNumber": t, "isDuringSearch": False,
                    "rootInfo": {"currentPlayer": to_move, "winrate": win, "scoreLead": sign * lead_tm, "visits": sum(m["visits"] for m in infos)},
                    "moveInfos": infos, "ownership": [sign * o for o in own_tm]}
            print(json.dumps(resp), flush=True)
            if os.environ.get("FAKE_TEACHER_DUPLICATE"):
                print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
