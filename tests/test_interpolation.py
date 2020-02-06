from cho_util.math import transform as tx

def main():
    q = tx.rotation.quaternion.random()
    axa = tx.rotation.quaternion.to_axis_angle(q)

    w = axa[:3]
    theta = axa[-1]
