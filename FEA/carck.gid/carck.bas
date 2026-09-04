*nnodes *nelems

COORDINATES
*Loop Nodes
*NodesNum *NodesCoord(1,real) *NodesCoord(2,real)
*End Nodes
END_COORDINATES

ELEMENT
*Loop Elements
*ElemsNum *ElemsConec *ElemsMatNum
*End Elements
END_ELEMENT

MOMENT-LOAD
*Loop Lines
*if(ConditionExist(Moment-Load,line))
*Set var nlist *LineNodes
*Loop nlist
Node, *GlobalNodes(nlist), UX, *Cond(Moment-Load,X-Moment), UY, *Cond(Moment-Load,Y-Moment)
*End nlist
*endif
*End Lines
END_MOMENT-LOAD

PRESURE
*Loop Lines
*if(ConditionExist(Presure,line))
*Set var nlist *LineNodes
*Loop nlist
Node, *GlobalNodes(nlist), *Cond(Presure,Presure:)
*End nlist
*endif
*End Lines
END_PRESURE

DISPLACEMENT
*Loop Lines
*if(ConditionExist(Displacement,line))
*Set var nlist *LineNodes
*Loop nlist
Node, *GlobalNodes(nlist), *Cond(Displacement,UX), *Cond(Displacement,UY)
*End nlist
*endif
*End Lines
END_DISPLACEMENT

WALL
*Loop Lines
*if(ConditionExist(Wall,line))
*Set var nlist *LineNodes
*Loop nlist
Node, *GlobalNodes(nlist)
*End nlist
*endif
*End Lines
END_WALL

MATERIAL PROPERTIES
*Loop Materials
*MatNum *MatProp(YOUNG_(Ex),real) *MatProp(POISSON_(NUXY),real) *MatProp(TENSILE_STRENGTH_(SIGMA_T),real) *MatProp(FRACTURE_TOUGHNESS_(K_IC),real) *MatProp(DENSITY_(DENS),real)
*End Materials
END_MATERIAL PROPERTIES
