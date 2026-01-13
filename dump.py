# lattice height map stuff
# def calculate_height_map(self):
#    """
#    Tosses out the current height map and rebuilds it based on the current lattice state
#    """
#    cols = {}
#    for site in self.sites:
#        col_x, col_y = self._get_col_inx(site)
#        cols[(col_x, col_y)].append(site.id)
#        self.site_to_col[site.id] = (col_x, col_y)

#    # Sort the site ids in each column by z coordinate
#    for (col_x, col_y), site_ids in cols.items():
#        sorted_ids = sorted(site_ids, key=lambda sid: self.sites[sid].location[2])

#        self.col_to_sites[(col_x, col_y)] = sorted_ids

#        # Height goes by number of occupied sites, allows for vacancies and terraces
#        occupied = sum(
#            1 for sid in sorted_ids if self.sites[sid].occupation_type.value != 0
#        )

#        self.height_map[col_x, col_y] = occupied

# def _get_col_inx(self, site: LatticeSite):
#    """
#    This explicitly only works for a ZB lattice.
#    Returns the column of a lattice site relevant for the height map.
#    """
#    ix, iy, _ = site.cell
#    x, y, _ = site.location

#    sub_col_x = (x % 1) * 4
#    col_x = ix * 4 + sub_col_x

#    sub_col_y = (y % 1) * 4
#    col_y = iy * 4 + sub_col_y

#    return col_x, col_y
#
# self.site_to_col = {}
# self.col_to_sites = {}
# self.max_height = dimensions[2]
#  p
#
# TODO:
# Refactor methods in this class to avoid the implicit mutation of state

# The next lattice site in the z direction since there is no integer measure of location
# Spparks solves this problem by flying adatoms in from the top of the simulation area
# There are 8 cols in a unit cell among all of the lattices and sublattices
# In both x and y directions there will be some column of atoms every 0.25a'
#
# The entire point of the height map is for deposition events

# Once the lattice is built every lattice location has an id.
# The neighbor lists go by the lattice site id.
#
