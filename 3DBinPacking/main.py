from geometry import Container, Box
from constructive import ExtremePointsHeuristic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vizualizare_3d(container):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Bin Packing - Extreme Points")

    ax.set_xlim([0, container.width])
    ax.set_ylim([0, container.height])
    ax.set_zlim([0, container.depth])
    ax.set_xlabel('Axa X (Lățime)')
    ax.set_ylabel('Axa Y (Înălțime)')
    ax.set_zlabel('Axa Z (Adâncime)')

    culori = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

    for i, box in enumerate(container.placed_boxes):
        culoare = culori[i % len(culori)]
        x, y, z = box.x, box.y, box.z
        w, h, d = box.width, box.height, box.depth

        xx = [x, x, x+w, x+w, x]
        yy = [y, y+h, y+h, y, y]
        
        ax.plot(xx, yy, [z]*5, color=culoare, linewidth=2)
        ax.plot(xx, yy, [z+d]*5, color=culoare, linewidth=2)
        
        ax.plot([x, x], [y, y], [z, z+d], color=culoare, linewidth=2)
        ax.plot([x, x], [y+h, y+h], [z, z+d], color=culoare, linewidth=2)
        ax.plot([x+w, x+w], [y+h, y+h], [z, z+d], color=culoare, linewidth=2)
        ax.plot([x+w, x+w], [y, y], [z, z+d], color=culoare, linewidth=2)

    plt.show()

def main():
    print("Se pregătește containerul...")
    container = Container(10.0, 10.0, 10.0)

    euristica = ExtremePointsHeuristic(container)

    cutii_de_test = [
        Box(box_id=1, w=4, h=4, d=4),
        Box(box_id=2, w=5, h=5, d=5),
        Box(box_id=3, w=3, h=3, d=3),
        Box(box_id=4, w=6, h=2, d=4),
        Box(box_id=5, w=10, h=10, d=10)
    ]

    print("Începe plasarea cutiilor...")
    for cutie in cutii_de_test:
        succes = euristica.pack_box(cutie)
        if succes:
            print(f" Cutia {cutie.id} a fost plasată la: X={cutie.x}, Y={cutie.y}, Z={cutie.z}")
        else:
            print(f" Cutia {cutie.id} NU a încăput!")

    volum_ocupat = sum(b.width * b.height * b.depth for b in container.placed_boxes)
    volum_total = container.width * container.height * container.depth
    procent = (volum_ocupat / volum_total) * 100
    
    print("-" * 30)
    print(f"Grad de umplere: {procent:.2f}%")
    print(f"Cutii plasate: {len(container.placed_boxes)} din {len(cutii_de_test)}")
    
    vizualizare_3d(container)

if __name__ == "__main__":
    main()