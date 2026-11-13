"""Bounded local interface bijections, consistently directed from new labels to old."""
from dataclasses import dataclass, field
from itertools import islice
from numbers import Integral

from ..core.game import Scenario


class RelabellingBudgetExceeded(ValueError):
    """A declared map or table-expansion budget was exceeded."""


def _positive_budget(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _map_count(source, limit):
    """Bound n+sum(X_i)+sum(X_i*A_i) without materializing any map values."""
    count = source.parties
    if count > limit:
        raise RelabellingBudgetExceeded('Relabelling exceeds max_map_entries')
    for inputs, outputs in zip(source.input_sizes, source.output_sizes):
        if inputs > limit-count:
            raise RelabellingBudgetExceeded('Relabelling exceeds max_map_entries')
        count += inputs
        if inputs > (limit-count)//outputs:
            raise RelabellingBudgetExceeded('Relabelling exceeds max_map_entries')
        count += inputs*outputs
    return count


def _bounded(values, expected, name):
    entries = tuple(islice(iter(values), expected+1))
    if len(entries) != expected:
        raise ValueError(f'{name} has the wrong number of entries')
    return entries


def _bijection(values, size, name):
    values = _bounded(values, size, name)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size
           for value in values) or len(set(values)) != size:
        raise ValueError(f'{name} must be an integer bijection of its alphabet')
    return tuple(int(value) for value in values)


def _profile(values, sizes, name):
    values = _bounded(values, len(sizes), name)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size
           for value, size in zip(values, sizes)):
        raise ValueError(f'{name} contains a label outside its alphabet')
    return tuple(int(value) for value in values)


def _inverse(permutation):
    result = [0]*len(permutation)
    for new, old in enumerate(permutation):
        result[old] = new
    return tuple(result)


@dataclass(frozen=True, init=False)
class Relabelling:
    """An immutable interface isomorphism; all maps send target labels to source labels.

    pi[i] identifies the source party assigned to target party i. Input maps
    tau_i[x_new] and output maps sigma_i[x_new][a_new] are local bijections.
    The target cardinalities are derived from pi. A party renaming here is not
    a physical interparty swap channel, and resource assumptions must travel
    separately through the same renaming.
    """
    source: Scenario
    target: Scenario = field(init=False)
    party_permutation: tuple[int, ...]
    input_maps: tuple[tuple[int, ...], ...]
    output_maps: tuple[tuple[tuple[int, ...], ...], ...]
    max_map_entries: int = field(default=1_000_000, repr=False, compare=False)

    def __init__(self, source, party_permutation=None, input_maps=None, output_maps=None,
                 *, max_map_entries=1_000_000):
        if not isinstance(source, Scenario):
            raise TypeError('Relabelling source must be a Scenario')
        limit = _positive_budget(max_map_entries, 'max_map_entries')
        _map_count(source, limit)
        count = source.parties
        parties = tuple(range(count)) if party_permutation is None else _bijection(
            party_permutation, count, 'party_permutation')
        target = Scenario(tuple(source.input_sizes[i] for i in parties),
                          tuple(source.output_sizes[i] for i in parties))
        if input_maps is None:
            inputs = tuple(tuple(range(size)) for size in target.input_sizes)
        else:
            inputs = tuple(_bijection(values, size, 'input map') for values, size in zip(
                _bounded(input_maps, count, 'input_maps'), target.input_sizes))
        if output_maps is None:
            outputs = tuple((tuple(range(answers)),)*questions for questions, answers in
                            zip(target.input_sizes, target.output_sizes))
        else:
            outputs = tuple(tuple(_bijection(values, answers, 'output map')
                                  for values in _bounded(party, questions, 'output input axis'))
                            for party, questions, answers in zip(
                                _bounded(output_maps, count, 'output_maps'),
                                target.input_sizes, target.output_sizes))
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'party_permutation', parties)
        object.__setattr__(self, 'input_maps', inputs)
        object.__setattr__(self, 'output_maps', outputs)
        object.__setattr__(self, 'max_map_entries', limit)

    def source_input(self, target_input):
        values = _profile(target_input, self.target.input_sizes, 'target input')
        result = [0]*self.source.parties
        for party, old in enumerate(self.party_permutation):
            result[old] = self.input_maps[party][values[party]]
        return tuple(result)

    def source_output(self, target_input, target_output):
        inputs = _profile(target_input, self.target.input_sizes, 'target input')
        outputs = _profile(target_output, self.target.output_sizes, 'target output')
        result = [0]*self.source.parties
        for party, old in enumerate(self.party_permutation):
            result[old] = self.output_maps[party][inputs[party]][outputs[party]]
        return tuple(result)

    def inverse(self, *, max_map_entries=None):
        """Invert the isomorphism, including the input-dependent output-map index."""
        limit = self.max_map_entries if max_map_entries is None else _positive_budget(max_map_entries, 'max_map_entries')
        _map_count(self.target, limit)
        parties = _inverse(self.party_permutation)
        inputs = tuple(_inverse(self.input_maps[old]) for old in parties)
        outputs = tuple(tuple(_inverse(self.output_maps[old][new_input]) for new_input in inputs[party])
                        for party, old in enumerate(parties))
        return Relabelling(self.target, parties, inputs, outputs, max_map_entries=limit)

    def then(self, next_map, *, max_map_entries=None):
        """Compose self:S->T followed by next_map:T->U, returning S->U."""
        if not isinstance(next_map, Relabelling):
            raise TypeError('Composition requires another Relabelling')
        if next_map.source != self.target:
            raise ValueError('Relabelling composition has incompatible intermediate interfaces')
        limit = min(self.max_map_entries, next_map.max_map_entries) if max_map_entries is None else _positive_budget(
            max_map_entries, 'max_map_entries')
        _map_count(self.source, limit)
        parties = tuple(self.party_permutation[i] for i in next_map.party_permutation)
        inputs = tuple(tuple(self.input_maps[old][value] for value in next_map.input_maps[party])
                       for party, old in enumerate(next_map.party_permutation))
        outputs = tuple(tuple(tuple(self.output_maps[old][next_map.input_maps[party][question]][value]
                                    for value in output_map)
                              for question, output_map in enumerate(next_map.output_maps[party]))
                        for party, old in enumerate(next_map.party_permutation))
        return Relabelling(self.source, parties, inputs, outputs, max_map_entries=limit)
