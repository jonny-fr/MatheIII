import sympy as sp

# Variablen und Funktion definieren
x, y = sp.symbols('x y', real=True)
f = x ** 3 + y ** 3 - 3 * x * y


def analyze_critical_points(f, vars):
    """Analysiert kritische Punkte: Gradient → Lösen → Hessian → Klassifikation"""

    # 1. Gradient berechnen
    grad = [sp.diff(f, v) for v in vars]

    # 2. Kritische Punkte (nur reelle)
    crit_points = sp.solve(grad, vars, dict=True)

    # 3. Hessian-Matrix
    H = sp.hessian(f, vars)

    print("∇f =", grad)
    print("\nKritische Punkte:", crit_points if crit_points else "Keine")
    print("\nHessian:\n", H)

    # 4. Klassifikation
    print("\n" + "=" * 60)
    for pt in crit_points:
        H_pt = H.subs(pt)
        det_H = H_pt.det()

        if len(vars) == 2:
            # 2D: Determinanten-Test
            a11 = H_pt[0, 0]
            if det_H > 0:
                typ = "lokales Minimum" if a11 > 0 else "lokales Maximum"
            elif det_H < 0:
                typ = "Sattelpunkt"
            else:
                typ = "unbestimmt"
        else:
            # n-D: Eigenwerte
            evs = list(H_pt.eigenvals().keys())
            if all(ev > 0 for ev in evs):
                typ = "lokales Minimum"
            elif all(ev < 0 for ev in evs):
                typ = "lokales Maximum"
            elif all(ev != 0 for ev in evs):
                typ = "Sattelpunkt"
            else:
                typ = "unbestimmt"

        print(f"\nPunkt: {pt}")
        print(f"  H = {H_pt},  det(H) = {det_H}")
        print(f"  → {typ}")


# Ausführen
analyze_critical_points(f, [x, y])
