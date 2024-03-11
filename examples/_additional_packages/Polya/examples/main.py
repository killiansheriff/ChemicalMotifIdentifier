from polya import Polya

pl = Polya(graph_name="fcc")
p_g, nms = pl.get_gt(ntypes=3)
print(p_g)
