
Classes = {}
def parser(type = None, name = None):
	def _parser(type):
		_name = name
		if _name is None:
			_name = type.__name__[5:]
		Classes[_name] = type
		return type

	if type is None:
		return _parser
	else:
		return _parser(type)

@parser
def parseLidl(img, labels, ret):
	pass

@parser
def parseKarstadt(img, labels, ret):
	pass
